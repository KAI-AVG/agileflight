import os
from os.path import join as opj
import sys
import yaml
import rospy
import numpy as np
from datetime import datetime
from dodgeros_msgs.msg import QuadState
from envsim_msgs.msg import ObstacleArray
from std_msgs.msg import Empty

from uniplot import plot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
base_path=os.environ.get("project_path")
from omegaconf import OmegaConf,DictConfig
sys.path.append(opj(base_path,"utils"))
from logger import Logger
import hydra

class Evaluator:
    def __init__(self, config):
        self.logger = Logger(name="Evaluator")
        self.logger.info("Initializing evaluator...")
        rospy.init_node("evaluator", anonymous=False)
        self.config = config
        self.base_path=base_path
        self.exp_name = config.exp_name
        self.target = config.target
        self.quad_radius = config.run.quad_radius
        self.timeout = config.evaluation.timeout
        self._initSubscribers(config.evaluation.topics)
        self._initPublishers(config.evaluation.topics)
        self.bounding_box = np.reshape(
            np.array(config.evaluation.bounding_box, dtype=float), (3, 2)
        ).T 
        self.is_active = False
        self.plots=config.plots
        self.pos = []
        self.dist = []
        self.time_array = (self.target + 1) * [np.nan]
        self.summary_path = None
        if config.mode == 'vision':
            self.input_mode = config.input_mode
            batch_size = config.batch_size
            dataset_name = f'{config.level}_{config.ob_type}_{config.desiredVel}'
            assert self.input_mode in ("depth","dvs","fusion"),f"input_mode must be in depth,dvs or fusion"
            self.model_name = config.model._target_.split('.')[-1]
            path = f'{self.model_name}_{self.input_mode}_{dataset_name}_b{batch_size}'
            rnn_type = config.model.get('rnn_type') if 'rnn_type' in config.model and config.model.rnn_type is not None else ''
            if rnn_type != '':
                assert rnn_type in ['lstm','rnn','gru'],'rnn must be implemented'
                path += f'_{rnn_type}'
            has_state = config.model.get('has_state') if 'rnn_type' in config.model else ''
            if has_state:
                path += f'_state'
            cross_mode = config.model.get('cross_mode') if 'cross_mode' in config.model else ''
            if cross_mode != '':
                path +=f'_{cross_mode}'
            summary_path = opj(f'experiment{config.experiment_num}','logs',path,f'checkpoint_epoch_{config.checkpoint_epoch}')
            self.summary_path = opj(base_path,summary_path)
            os.makedirs(self.summary_path,exist_ok=True)
            config_path = opj(self.summary_path, 'config.yaml')
            if not os.path.exists(config_path):
                OmegaConf.save(config=config, f=config_path)
        self.hit_obstacle = False
        self.crash = 0
        self.in_crash = 0
        self.ctr = 0
        self.start_time_mark = False

        # if trees environment, collision checking should only consider the x, y position of the drone (assume perfectly vertical, cylindrical trunks)
        with open(opj(base_path,'flightmare/flightpy/configs/vision/config.yaml')) as sim_config:
            config_params = yaml.load(sim_config, Loader=yaml.FullLoader)
        self.is_trees = 'trees' in config_params['environment']['level']
        self.logger.debug(f'is_trees = {self.is_trees}')

    def _initSubscribers(self, config):
        self.state_sub = rospy.Subscriber(
            "/%s/%s" % (config.quad_name, config.state),
            QuadState,
            self.callbackState,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.obstacle_sub = rospy.Subscriber(
            "/%s/%s" % (config.quad_name, config.obstacles),
            ObstacleArray,
            self.callbackObstacles,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.start_sub = rospy.Subscriber(
            "/%s/%s" % (config.quad_name, config.start),
            Empty,
            self.callbackStart,
            queue_size=1,
            tcp_nodelay=True,
        )

    def _initPublishers(self, config):
        self.finish_pub = rospy.Publisher(
            "/%s/%s" % (config.quad_name, config.finish),
            Empty,
            queue_size=1,
            tcp_nodelay=True,
        )

    def publishFinish(self):
        self.finish_pub.publish()
        self.writeSummary()
        self.printSummary()

    def callbackState(self, msg):

        self.pos_x = msg.pose.position.x

        # mark start time based on position rather than start signal
        if self.pos_x > 0.5 and not self.start_time_mark:
            self.time_array[0] = rospy.get_rostime().to_sec()
            self.start_time_mark = True

        # self.ctr += 1
        # if self.ctr % 30 != 0:
        #     print(f'[evaluator] self.is_active={self.is_active} ; self.time_array[0]={self.time_array[0]:.3f}')

        if not self.is_active:
            return

        pos = np.array(
            [
                msg.header.stamp.to_sec(),
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ]
        )
        self.pos.append(pos)

        bin_x = int(max(min(np.floor(self.pos_x), self.target), 0))
        if np.isnan(self.time_array[bin_x]):
            self.time_array[bin_x] = rospy.get_rostime().to_sec()

        if self.pos_x > self.target:
            self.is_active = False
            self.publishFinish()

        if rospy.get_time() - self.time_array[0] > self.timeout:
            self.abortRun()

        outside = ((pos[1:] > self.bounding_box[1, :]) | (pos[1:] < self.bounding_box[0, :])
        ).any(axis=-1)
        if (outside == True).any():
            self.abortRun()

    # Note, the start signal may need to be sent multiple times. Sometimes once doesn't work.
    # So, self.time_array[0] is set in callbackState based on position instead.
    def callbackStart(self, msg):
        if not self.is_active:
            self.is_active = True
        # self.time_array[0] = rospy.get_rostime().to_sec()

    def callbackObstacles(self, msg):
        if not self.is_active:
            return

        # sample closest obstacle and do collision checking
        obs = msg.obstacles[0]
        # if self.is_trees:
        #     dist = np.linalg.norm(np.array([obs.position.x, obs.position.y]))
        #     # print(f'[EVALUATOR_NODE] obst x, y = [{obs.position.x:.2f}, {obs.position.y:.2f}]')
        # else:
        dist = np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z]))
        # margin is distance to object center minus object radius minus drone radius (estimated)
        margin = dist - obs.scale
        #print(f'[EVALUATOR_NODE] margin = {margin:.2f}')
        self.dist.append([msg.header.stamp.to_sec(), margin])
        if margin < 0:
            if not self.hit_obstacle:
                self.crash += 1
                self.logger.info(f"Crashed (quadrotor radius)!")
                if dist**2 + self.quad_radius**2 < obs.scale**2:
                    self.in_crash +=1
            self.hit_obstacle = True
        else:
            self.hit_obstacle = False

    def abortRun(self):
        self.logger.info("You did not reach the goal!")
        summary = {}
        summary["Success"] = False
        if self.summary_path:
            summary_file_path = opj(self.summary_path, "summary.yaml")
            with open(summary_file_path, "a") as f:
                yaml.safe_dump(summary, f)
        else:
            with open("summary.yaml", "a") as f:
                yaml.safe_dump(summary, f)
        rospy.signal_shutdown("Completed Evaluation")

    def writeSummary(self):
        """
        - second was logging the whole path.
        """
        #Time taken throughout the run
        self.timeTaken = self.time_array[-1] - self.time_array[0]

        #Distance from nearest obstacles
        dist = np.array(self.dist) #-> time, distance

        #Whole path
        pos = np.array(self.pos) #-> time, x,y,z

        #Since the size of position and nearest obstacle is different, we can't append to same df
        #We also shouldn't interpolate -> can harm the data
        #Saving to two different csv because one's frequency is double than other
        if self.summary_path:
            self.exp_dir = opj(self.summary_path,"stored_metrics",self.exp_name)
        else:
            self.exp_dir = opj(self.base_path,"envtest/stored_metrics",self.exp_name)
        os.makedirs(self.exp_dir,exist_ok=True)

        #XYZ Path File
        # print(pos)
        # print(pos.shape)
        pathFile = os.path.join(self.exp_dir,"path.csv")
        pd.DataFrame(pos).to_csv(pathFile)

        pathPlots = os.path.join(self.exp_dir,"XYZ Plots.png")
        _, axs = plt.subplots(3, 1, figsize=(16, 20))
        pos = pos.T
        axs[0].plot(pos[1],pos[2])
        axs[0].set_visible(False)
        axs[0].set_xlabel("X [m]")
        axs[0].set_ylabel("Y [m]")
        axs[0].set_title("TOP-DOWN; XY")

        axs[1].plot(pos[2], pos[3])
        axs[0].set_visible(False)
        axs[1].set_xlabel("Y [m]")
        axs[1].set_ylabel("Z [m]")
        axs[1].set_title("HEAD-ON; YZ")
        axs[1].invert_xaxis()
        
        axs[2].plot(pos[1], pos[3])
        axs[0].set_visible(False)
        axs[2].set_xlabel("X [m]")
        axs[2].set_ylabel("Z [m]")
        axs[2].set_title("SIDE-VIEW; ZX")

        plt.savefig(pathPlots)

        #Distance to Obstacle File
        distFile = os.path.join(self.exp_dir,"dist.csv")
        nearestDistPlots = os.path.join(self.exp_dir,"nearestDist.png")
        pd.DataFrame(dist).to_csv(distFile)
        plt.figure()
        plt.plot(dist[:, 0] - self.time_array[0], dist[:, 1])
        plt.xlabel("time (s)")
        plt.ylabel("Distance from Obstacles [m]")
        plt.grid()
        plt.savefig(nearestDistPlots)

        # save trainset folder name so more stats can be extracted later
        subdirs = sorted(os.listdir(opj(base_path,"envtest/rollouts")))
        stats_dir = subdirs[-1]

        #Time Taken and num collisions Dat File
        Scalarfile = os.path.join(self.exp_dir,"scalarMetrics.dat")
        with open(Scalarfile, "a") as file:
            file.write(str( float(self.timeTaken) ) + ", " + str(int(self.crash)) + ", " + stats_dir + "\n")


    def printSummary(self):
        
        ttf = self.time_array[-1] - self.time_array[0]
        summary = {}
        summary["Success"] = True if self.crash == 0 else False
        self.logger.info("You reached the goal in %5.3f seconds" % ttf)
        summary["time_to_finish"] = ttf
        self.logger.info("Your intermediate times are:")
        print_distance = 10
        summary["segment_times"] = {}
        for i in range(print_distance, self.target + 1, print_distance):
            self.logger.info("    %2i: %5.3fs " % (i, self.time_array[i] - self.time_array[0]))
            summary["segment_times"]["%i" % i] = self.time_array[i] - self.time_array[0]
        self.logger.info("You hit %i obstacles" % self.crash)
        summary["number_crashes"] = self.crash
        summary["number_in_crashes"] = self.in_crash
        if self.summary_path:
            summary_file_path = opj(self.summary_path, "summary.yaml")
            with open(summary_file_path, "a") as f:
                yaml.safe_dump(summary, f)
        else:
            with open("summary.yaml", "a") as f:
                yaml.safe_dump(summary, f)
        if not self.plots:
            rospy.signal_shutdown("Completed Evaluation")
            return

        self.logger.debug("Here is a plot of your trajectory in the xy plane")
        pos = np.array(self.pos)
        plot(xs=pos[:, 1], ys=pos[:, 2], color=True)

        self.logger.debug("Here is a plot of your average velocity per 1m x-segment")
        x = np.arange(1, self.target + 1)
        dt = np.array(self.time_array)
        y = 1 / (dt[1:] - dt[0:-1])
        plot(xs=x, ys=y, color=True)

        self.logger.debug("Here is a plot of the distance to the closest obstacles")
        dist = np.array(self.dist)
        plot(xs=dist[:, 0] - self.time_array[0], ys=dist[:, 1], color=True)

        rospy.signal_shutdown("Completed Evaluation")


@hydra.main(config_path=opj(base_path, "configs"), config_name="sim_config",version_base="1.3")
def main(cfg: DictConfig):
    # experiment name is passed in as argument in batched rollouts,
    # otherwise it is the current datetime
    cfg.exp_name = datetime.now().strftime('%Y-%m-%d--%H_%M_%S')
    Evaluator(cfg)
    rospy.spin()

if __name__ == "__main__":
    main()



