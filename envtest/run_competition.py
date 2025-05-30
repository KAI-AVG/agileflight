# %%
#!/usr/bin/python3
from datetime import datetime
import shutil
import rospy
from std_msgs.msg import Empty, String, Header
from geometry_msgs.msg import Vector3, TwistStamped
from sensor_msgs.msg import Image
from dodgeros_msgs.msg import Command, QuadState
from envsim_msgs.msg import ObstacleArray, Obstacle
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from user_code import compute_command_state_based
from utils import AgileCommandMode, AgileQuadState, AgileCommand
import time
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
import cv2
import torch
import yaml
base_path2=os.environ.get("project_path")
sys.path.append(opj(base_path2,"event_denoise"))
SMALL_EPS = 1e-5
sys.path.append(opj(base_path2,'learner')) # type: ignore
import policy_model
sys.path.append(opj(base_path2,'utils')) # type: ignore
from logger import Logger
from ev_utils import * 
from omegaconf import OmegaConf,DictConfig
import esim_torch
from scipy.signal import convolve2d
import hydra
from hydra.utils import instantiate 


# %%
class AgilePilotNode:
    def __init__(self,config:DictConfig):
        self.logger=Logger(name="RUN")
        self.logger.info("Initializing agile_pilot_node...")
        rospy.init_node("agile_pilot_node", anonymous=False)
        self.config=config
        self.resize_input=config.run.resize_input
        self.data_buffer_maxlength =config.run.data_buffer_maxlength
        self.exp_name = config.exp_name
        self.total_num_exps = config.total_num_exps
        self.desiredVel=config.desiredVel
        self.target=config.target
        self.mode = config.mode
        assert self.mode == 'vision' or self.mode =='expert'
        self.do_events = self.mode =="vision"
        self.plot_cmd = False
        self.extras = None
        quad_name = config.evaluation.topics.quad_name
        if self.mode == 'vision':
            self.input_mode = config.input_mode
            batch_size = config.batch_size
            dataset_name = f'{config.level}_{config.ob_type}_{config.desiredVel}'
            assert self.input_mode in ("depth","dvs","fusion"),f"input_mode must be in depth,dvs or fusion"
            self.Model = instantiate(config.model)
            self.model_name = config.model._target_.split('.')[-1]
            if "Fusion" in self.model_name:
                assert self.input_mode == "fusion","input mode must be fusion"
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
            checkpoint_path = opj(f'experiment{config.experiment_num}','models',path,f'checkpoint_epoch_{config.checkpoint_epoch}')
            #self.Model=RVT_Stack(is_depth=self.is_depth,dvs_num=self.dvs_num)
            model_path = opj(base_path2,f'{checkpoint_path}.pth')
            if not os.path.exists(model_path):
                checkpoint_path = opj(f'experiment{config.experiment_num}','models',path,f'checkpoint_epoch_best_{config.checkpoint_epoch}')
                #self.Model=RVT_Stack(is_depth=self.is_depth,dvs_num=self.dvs_num)
                model_path = opj(base_path2,f'{checkpoint_path}.pth')
            assert os.path.exists(model_path)
            checkpoint = torch.load(model_path)
            self.Model.load_state_dict(checkpoint['model_state_dict'])
            # if self.checkpoint_path is not None:
            #     self.Model.load_from_checkpoint(self.checkpoint_path,self.combine_checkpoints)
            self.Model.eval()
            # Initialize hidden state
            #self.Model.model_hidden_state=[None]
        self.esim = esim_torch.ESIM(contrast_threshold_neg=config.run.neg_thred,
                            contrast_threshold_pos=config.run.pos_thred,
                            refractory_period_ns=.5e6) 
        #######################

        ########################################
        ## Set up NN and other configurations ##
        ########################################

        # load yaml of parameters
        with open(opj(base_path2,'envsim/parameters/simple_sim_pilot.yaml')) as file: # type: ignore
            pilot_params = yaml.load(file, Loader=yaml.FullLoader)
            self.takeoff_height = pilot_params['takeoff_height']

        with open(opj(base_path2,'flightmare/flightpy/configs/vision/config.yaml')) as file: # type: ignore
            config_params = yaml.load(file, Loader=yaml.FullLoader)
            camera_params = config_params['rgb_camera']

        self.image_h, self.image_w = (camera_params['height'], camera_params['width'])
        # self.gimbal_h, self.gimbal_w = (60, 90)
        # self.gimbal_fov = camera_params['fov']
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        
        # logging
        self.init = 0
        self.col = 0
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        # self.last_valid_im = None #Image that will be logged
        self.data_format = {'timestamp':[],
                            'desired_vel':[],
                            'quat_1':[],
                            'quat_2':[],
                            'quat_3':[],
                            'quat_4':[],
                            'pos_x':[],
                            'pos_y':[],
                            'pos_z':[],
                            'vel_x':[],
                            'vel_y':[],
                            'vel_z':[],
                            'velcmd_x':[],
                            'velcmd_y':[],
                            'velcmd_z':[],
                            'ct_cmd':[],
                            'br_cmd_x':[],
                            'br_cmd_y':[],
                            'br_cmd_z':[],
                            'obstacle_distance':[],
                            'is_collide': [],
                            }
        self.data_buffer = pd.DataFrame(self.data_format) # store in the data frame
        self.log_ctr = 0 # counter for the csv, unused for now
        # if goal distance is 60, end of data collection xrange is 50
        self.data_collection_xrange = [0+5, self.target-.17*self.target]
        # make the folder for the epoch
        self.folder = opj(base_path2,"envtest/rollouts",self.exp_name) # type: ignore
        os.makedirs(self.folder,exist_ok=True)
        # if this is a named experiment, save the config file to maintain information of run, including scene/env/etc
        shutil.copy(opj(base_path2, 'configs/sim_config.yaml'), opj(self.folder, 'simulation_config.yaml')) 
        shutil.copy(opj(base_path2, 'flightmare/flightpy/configs/vision/config.yaml'), opj(self.folder, 'competition_config.yaml')) 
        self.events = np.zeros((4,self.image_h, self.image_w))

        # if save_events, save each event frame via the log function and then save as a npy
        self.evims = []
        self.im_dbg2s = []
        self.state_poss = []
        self.state_vels = []
        self.expert_command = None
        self.expert_commands = []
        self.vision_commands = []
        self.spline_poss = []
        self.spline_vels = []
        self.plotted_commands = False
        self.start_time = 0
        self.logged_time_flag = 0
        self.first_data_write = False
        self.current_cmd_controller = None
        self.current_cmd = None
        self.keyboard_input = ''
        self.got_keypress = 0.0
        # initialize to bogus obstacle array with 10 obstacles at 1000, 1000, 1000
        self.obs_msg = self.create_obstacle_array()
        # vision member variables
        self.depth = np.zeros((self.image_h, self.image_w))
        self.depth_t = None
        self.depth_im_threshold = 0.9 # increased from 0.1 (max depth seems to be ~ 0.885)
        self.im = np.zeros((self.image_h, self.image_w),dtype=np.float32)
        self.im_t = 0
        self.im_ctr = 0
        self.prev_im = np.zeros((self.image_h, self.image_w),dtype=np.float32)

        # self.depth_gimbal = np.zeros((self.gimbal_h, self.gimbal_w))
        # self.im_gimbal = np.zeros((self.gimbal_h, self.gimbal_w))
        # self.prev_im_gimbal = np.zeros((self.gimbal_h, self.gimbal_w))
        # self.gimbal = None
        self.pts = [[0,0], [0,0], [0,0], [0,0]]
        self.im_dbg1 = None
        self.im_dbg2 = None
        # place to store extras from learned model
        self.extras = None
        # manual synchronization variables
        self.accepted_delta_t_im_depth = 0.01

        self.csv_file = base_path2+'/flightmare/flightpy/configs/vision/'+config_params['environment']['level']+'/'+config_params['environment']['env_folder']+'/static_obstacles.csv' # type: ignore
        self.is_trees = 'trees' in config_params['environment']['level'] or 'forest' in config_params['environment']['level']

        #####################
        ## ROS subscribers ##
        #####################

        # Logic subscribers
        self.start_sub = rospy.Subscriber(
            "/" + quad_name + "/start_navigation",
            Empty,
            self.start_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Observation subscribers
        # we are making odom, image, and depth approximately time synchronized for logging purposes
        self.odom_sub = message_filters.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/state",
            QuadState,
        )
        self.im_sub = message_filters.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/image",
            Image,
        )
        self.depth_sub = message_filters.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/unity/depth",
            Image,
        )
        timesync = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.im_sub, self.depth_sub], queue_size=10, slop=self.accepted_delta_t_im_depth)
        timesync.registerCallback(self.observation_callback)

        self.obstacle_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles",
            ObstacleArray,
            self.obstacle_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.cmd_sub = rospy.Subscriber(
            "/" + quad_name + "/dodgeros_pilot/command",
            Command,
            self.cmd_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        ####################
        ## ROS publishers ##
        ####################

        # Command publishers
        self.cmd_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/feedthrough_command",
            Command,
            queue_size=1,
        )
        self.linvel_pub = rospy.Publisher(
            "/" + quad_name + "/dodgeros_pilot/velocity_command",
            TwistStamped,
            queue_size=1,
        )
        self.im_dbg1_pub = rospy.Publisher(
            "/debug_img1",
            Image,
            queue_size=1,
        )
        self.im_dbg2_pub = rospy.Publisher(
            "/debug_img2",
            Image,
            queue_size=1,
        )
        self.logger.info("Initialization completed!")


    #############################
    ## Vision-based controller ##
    #############################

    def compute_command_vision_based(self):

        # Example of LINVEL command (velocity is expressed in world frame)
        command_mode = 2
        command = AgileCommand(command_mode)
        if self.state is None:
            self.logger.error("state is None")
        command.t = self.state.t # type: ignore
        command.yawrate = 0.0
        command.mode = command_mode
        
        ###############
        ## Load data ##
        ###############
        # determine model input image
        if self.input_mode == "depth":
            depth_input = torch.from_numpy(self.depth.astype("float32")).unsqueeze(0).unsqueeze(0)
            depth_input = torch.nn.functional.interpolate(depth_input, size=(60, 90), mode='bilinear', align_corners=False).squeeze().numpy()
            # if self.motion_blur:
            #     depth_input = self.add_motion_blur(depth_image=depth_input,speed=self.desiredVel,motion_size=self.motion_blur)
            inputs = (torch.from_numpy(depth_input).unsqueeze(0).unsqueeze(0).float(),)
        elif self.input_mode == "dvs":
            inputs = (torch.tensor(self.events).unsqueeze(0).float(),)
        else:
            depth_input = torch.from_numpy(self.depth.astype("float32")).unsqueeze(0).unsqueeze(0)
            depth_input = torch.nn.functional.interpolate(depth_input, size=(60, 90), mode='bilinear', align_corners=False).squeeze().numpy()
            depth_input = torch.from_numpy(depth_input).unsqueeze(0).unsqueeze(0).float()
            dvs_input = torch.tensor(self.events).unsqueeze(0).float()
            inputs = (dvs_input,depth_input)
        state = np.concatenate([self.state.att,self.state.pos,self.state.vel],axis=0)
        state = torch.tensor(state).unsqueeze(0)
        # if self.do_events:
        #     # set this by the percentile
        #     im_scaledown_factor = torch.quantile(torch.abs(im), 0.97)
        # else:
        #     im_scaledown_factor = 1.0
        with torch.no_grad():
            x,self.extras,others = self.Model(*inputs,state,extras=self.extras)
            # hidden state that for combo origunet+X model should be an unraveled iterable of ((origunet_unet_hidden, origunet_velpred_hidden, X_hidden
        #self.Model.update_hidden_state(self.extras)
        # print(f'[RUN_COMPETITION VISION_BASED] model output {x}')

        #x = x.squeeze().detach().numpy()
        x=x.detach().squeeze(0).numpy()
        vx, vy = x[0],x[1]
        com = np.array([vx,vy,0])*self.desiredVel
        command.velocity = com
        # possibly necessary scalers if using a pretrained V(phi) from another environment
        # command.velocity[1] *= 2.0

        # manual drone acceleration phase
        min_xvel_cmd = 1.0
        hardcoded_ctl_threshold = 2.0
        if self.state.pos[0] < hardcoded_ctl_threshold: # type: ignore
            command.velocity[0] = max(min_xvel_cmd, (self.state.pos[0]/hardcoded_ctl_threshold)*self.desiredVel) # type: ignore
        
        return command

    def cmd_callback(self, msg):
        self.current_cmd_controller = msg

    # legacy
    def readVel(self,file):
        with open(file,"r") as f:
            x = f.readlines()
            for i in range(len(x)):
                if i == 0:
                    return float(x[i].split("\n")[0])
    # compute estimated events from two stored images, with thresholds inputted
    # network was trained from evims of floats binned by 0.2, so estimate that here
    def compute_events(self, neg_thresh=0.2, pos_thresh=0.2, gimbal=False):

        im = self.im
        prev_im = self.prev_im
        h = self.image_h
        w = self.image_w

        events_zero = np.zeros((2,60,90))
        
        if im is None or prev_im is None:
            self.events = events_zero
            return
        
        device = torch.device("cuda")
        img_seq = np.stack([prev_im,im],axis=0)
        img_seq_tensor = torch.from_numpy(img_seq.astype("float32")).unsqueeze(0).cpu() 
        img_seq_resized = torch.nn.functional.interpolate(img_seq_tensor, size=(60, 90), mode='bilinear', align_corners=False).cpu()
        img_seq_resized = img_seq_resized.squeeze(0).numpy()
        log_img = torch.from_numpy(np.log(img_seq.astype("float32") + 1e-10)).to(device)
        timestapes=np.array([self.prev_t,self.im_t])
        timestamps_ns = torch.from_numpy((timestapes * 1e9).astype("int64")).to(device)
        events = self.esim.forward(log_img,timestamps_ns)
        for k in events.keys():
            events[k]=events[k].cpu()
        log_img = log_img.cpu()
        timestamps_ns = timestamps_ns.cpu()
        ts = events['t']
        event_indices = torch.bitwise_and((ts >= self.prev_t), (ts <= self.im_t))
        x, y, t, p = torch.stack([events[key][event_indices].unsqueeze(1) for key in ['x', 'y', 't', 'p']], dim=0).unbind(0)
        event_window = torch.cat((x,y,t,p),dim=1)
        sorted_indices = torch.argsort(event_window[:, 2]) 
        event_array = event_window[sorted_indices].numpy()
        # if event_window.shape[0]>1:
        #         if self.denoising == 'mlpfilter':
        #             from mlpffilteringEvents import mlpffiltering # type: ignore
        #             event_window = event_window.numpy()
        #             event_window = event_window[:,[0,1,3,2]]
        #             event_window[:,3] = event_window[:,3]/1e6
        #             denoise_event = mlpffiltering(event_window,0.5,"ds")
        #             event_window[:,3] = event_window[:,3]*1e6
        #             denoise_event = denoise_event[:,[0,1,3,2]]
        #             event_window = torch.tensor(denoise_event)
        #         elif self.denoising == "stcf":
        #             from stcfforshang import stcfforshang # type: ignore
        #             event_window = event_window.numpy()
        #             event_window = event_window[:,[2,0,1,3]]
        #             event_window[:,0] = event_window[:,0]/1e3
        #             denoise_event = stcfforshang(event_window)
        #             event_window[:,0] = event_window[:,0]*1e3
        #             denoise_event = denoise_event[:,[1,2,0,3]]
        #             event_window = torch.tensor(denoise_event)
        img_size = (60,90)
        evframes = np.zeros((2,*img_size))
        if event_array.shape[0]:
            t_norm = event_array[-1,2]-event_array[0,2]
            if t_norm != 0:
                event_array[:, 2] = event_array[:, 2] - event_array[0, 2]
                event_array[:, 2] = event_array[:, 2] / t_norm * 5
            else:
                event_array[:, 2] = 0
        else:
            self.events = events_zero
            return
        evframes[0, event_array[:, 1].astype(int), event_array[:, 0].astype(int)] += 1
        evframes[1, event_array[:, 1].astype(int), event_array[:, 0].astype(int)] = event_array[:, 2]
        if self.denoising=='connect':
            evframes = self.denoising(evframes=evframes,denoising=4)
        self.events = evframes
        # approximation of events calculation

        # difflog = np.log(im + SMALL_EPS) - np.log(prev_im + SMALL_EPS)

        # # thresholding
        # self.events = np.zeros_like(difflog)

        # if np.abs(difflog).max() < max(pos_thresh, neg_thresh):
        #     return

        # # quantize difflog by thresholds
        # pos_idxs = np.where(difflog > 0.0)
        # neg_idxs = np.where(difflog < 0.0)
        # self.events[pos_idxs] = (difflog[pos_idxs] // pos_thresh) * pos_thresh
        # self.events[neg_idxs] = (difflog[neg_idxs] // -neg_thresh) * -neg_thresh
        return
    
    # def denoising(self,evframes:np.array,denoising:int)->np.array:
    #     pos_img = (evframes[0]*10).astype(np.int8)
    #     retval,labels,stats,centroids = cv2.connectedComponentsWithStats(pos_img,connectivity=8)
    #     if retval >1:
    #         for i in range(1,retval):
    #             if stats[i,cv2.CC_STAT_AREA] < denoising:
    #                 labels[labels==i] =0
    #     for channel in [0, 2]:
    #         evframes[channel][labels == 0] = 0
    #     neg_img = (evframes[1]*10).astype(np.int8)
    #     retval,labels,stats,centroids = cv2.connectedComponentsWithStats(neg_img,connectivity=8)
    #     if retval >1:
    #         for i in range(1,retval):
    #             if stats[i,cv2.CC_STAT_AREA] < denoising:
    #                 labels[labels==i] =0
    #     for channel in [1,3]:
    #         evframes[channel][labels == 0] = 0
    #     return evframes
    
    # def add_motion_blur(self,depth_image, speed, motion_size)->np.array:
    #     kernel_size = int(speed * motion_size) 
    #     kernel = np.diag(np.ones(kernel_size))
    #     kernel = kernel / kernel_size
    #     # 使用numpy进行卷积操作
    #     blurred_image = convolve2d(depth_image, kernel, mode='same')
    #     blurred_image = np.clip(blurred_image, 0, 1).astype(depth_image.dtype)
    #     return blurred_image

    # approximate time-synced callback with three sensor measurements: odom state, rgb image, depth image
    def observation_callback(self, odom_msg, im_msg, depth_msg):

        ###################
        ### SUBSCRIBERS ###
        ###################

        # handle odom
        self.state_callback(odom_msg)

        # handle image
        if self.im_callback(im_msg) < 0:
            return

        # handle depth
        if self.depth_callback(depth_msg) < 0:
            return

        # run expert regardless of method
        if self.mode == 'expert':
        
            self.expert_command, extras = compute_command_state_based(
                state=self.state,
                obstacles=self.obs_msg,
                desiredVel=self.desiredVel,
                is_trees=self.is_trees,
                logger=self.logger
            )
            collisions = extras['collisions']
            wpt_idx = extras['wpt_idx']
            spline_poss = extras['spline_poss']
            spline_vels = extras['spline_vels']

        else:

            self.expert_command = None
            collisions = None
            wpt_idx = None
            spline_poss = None
            spline_vels = None

        # debug image 2; changeable debug image
        self.im_dbg2 = self.im.copy() # copying full image
        
        # if im_dgb2 is single-channel, make 3-channel
        if len(self.im_dbg2.shape) == 2:
            self.im_dbg2 = cv2.cvtColor((self.im_dbg2*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            # self.im_dbg2 = np.stack((self.im_dbg2,)*3, axis=-1)

        # if in vision command mode, compute vision command and publish
        vision_command = None
        if self.mode == 'vision':

            start_compute_time = time.time()
            vision_command = self.compute_command_vision_based()

            # useful occasional prints for debugging
            # if self.im_ctr % 20 == 0:
            #     self.logger.debug(f'compute_command_vision_based took {time.time() - start_compute_time:.3f} seconds')
            #     self.logger.debug(f'events min = {self.events.min():.2f}, events max = {self.events.max():.2f}, events 0.97 quantile = {torch.quantile(torch.abs(torch.Tensor(self.events)), 0.97):.2f}')

            #     self.logger.debug(f'depth min = {self.extras[0].min():.2f}, depth max = {self.extras[0].max():.2f}, depth 0.97 quantile = {torch.quantile(torch.abs(self.extras[0]), 0.97):.2f}')

            # if UNet type model, visualize the first element of extras which is fully interpolated up-to-size depth prediction from evframe
            # if self.Model.model_name == 'OrigUNet' or \
            # (isinstance(self.model_type, list) and self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'VITFLY_ViTLSTM') or \
            # (isinstance(self.model_type, list) and self.model_type[0] == 'OrigUNet' and self.model_type[1] == 'ConvNet_w_VelPred'):
                
            #     self.im_dbg2 = (np.stack((self.extras[0].squeeze().detach().numpy(),)*3, axis=-1) * 255).astype(np.uint8)
            
            # self.im_dbg2_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.im_dbg2, encoding="passthrough"))

            self.command = vision_command

        # if in state mode, compute state command and publish
        else:

            # user_code expert
            self.command = self.expert_command

            # debug image 2 will overlay the collision array of points as white dots,
            # where if collision[i, j] == 1 it is red,
            # and the wpt_idx as a green dot
            if collisions is not None:
                x_px_offset = self.im_dbg2.shape[1] / (collisions.shape[1]+1) # float
                y_px_offset = self.im_dbg2.shape[0] / (collisions.shape[0]+1) # float
                # collisions array goes from physical top left (body frame y=15, z=15) to bottom left
                # coordinates in waypoint frame
                for yi in range(collisions.shape[0]):

                    for xi in range(collisions.shape[1]):

                        if collisions[yi, xi] == 1:
                            color = (0, 0, 255) # red
                        else:
                            color = (255, 0, 0) # blue
                        pt_in = (int((xi+1)*x_px_offset), int((yi+1)*y_px_offset))
                        self.im_dbg2 = cv2.circle(self.im_dbg2, pt_in, 2, color, -1)

                # mark chosen waypoint with green circle
                if wpt_idx is not None:
                    pt_in_chosen = (int((wpt_idx[1]+1)*x_px_offset), int((wpt_idx[0]+1)*y_px_offset))
                    cv2.circle(self.im_dbg2, pt_in_chosen, 6, (0, 255, 0), -1)

        self.publish_command(self.command)

        # publish debug images
        # debug image 1; image and events overlayed + velocity command arrow

        im_dbg1 = self.depth.copy() if not self.do_events else self.im.copy()
        h, w = self.image_h, self.image_w

        # if self.do_events:

        #     im_dbg1_evs, enc = simple_evim(self.events, scaledown_percentile=.8, style='redblue-on-black') # copying cropped and horizon-aligned image
        #     # add in image for better visualization
        #     im_dbg1 = np.stack(((im_dbg1*255.0).astype(np.uint8),)*3, axis=-1)
        #     if self.events is not None:
        #         im_dbg1[np.where(self.events != 0.0)] = im_dbg1_evs[np.where(self.events != 0.0)]

        arrow_start = (w//2, h//2)
        if self.mode == 'vision' and wpt_idx is not None:
            arrow_end = pt_in_chosen
        else:
            arrow_end = (int(w/2-self.command.velocity[1]*(w/3)), int(h/2-self.command.velocity[2]*(h/3)))
        
        self.im_dbg1 = im_dbg1
        #self.im_dbg1 = cv2.arrowedLine( im_dbg1, arrow_start, arrow_end, (0, 0, 0), h//60, tipLength=0.2)
        self.im_dbg1_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.im_dbg1, encoding="passthrough"))

        self.im_dbg2 = cv2.arrowedLine( self.im_dbg2, arrow_start, arrow_end, (0, 0, 0), h//80, tipLength=0.15)
        self.im_dbg2_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.im_dbg2, encoding="bgr8"))

        # under some conditions, log sensor data
        # state, image, and depth image
        if self.state is not None and self.state.pos[0] > self.data_collection_xrange[0] and self.state.pos[0] < self.data_collection_xrange[1]:
            self.log_data(log_expert=False)
            self.plotted_commands = False
            self.expert_commands.append(self.expert_command)
            self.vision_commands.append(vision_command)
            self.spline_poss.append(spline_poss)
            self.spline_vels.append(spline_vels)
            self.state_vels.append(self.state.vel)
            self.state_poss.append(self.state.pos)

        # once the drone is beyond the collection range, save a plot of expert and vision commands
        if self.state is not None and self.state.pos[0] > self.data_collection_xrange[1] and not self.plotted_commands and self.plot_cmd:

            self.logger.debug(f'Plotting commands...')

            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(3, 2, figsize=(8, 8))

            axs[0, 0].plot([pos[0] for pos in self.spline_poss], label='spline pos') if spline_poss is not None else None
            axs[0, 0].plot([pos[0] for pos in self.state_poss], label='state pos')
            axs[0, 0].set_ylabel(f"x pos")
            axs[0, 0].legend()
            axs[0, 0].grid()

            axs[1, 0].plot([pos[1] for pos in self.spline_poss], label='spline pos') if spline_poss is not None else None
            axs[1, 0].plot([pos[1] for pos in self.state_poss], label='state pos')
            axs[1, 0].set_ylabel(f"y pos")
            axs[1, 0].legend()
            axs[1, 0].grid()

            axs[2, 0].plot([pos[2] for pos in self.spline_poss], label='spline pos') if spline_poss is not None else None
            axs[2, 0].plot([pos[2] for pos in self.state_poss], label='state pos')
            axs[2, 0].set_ylabel(f"z pos")
            axs[2, 0].legend()
            axs[2, 0].grid()

            axs[0, 1].plot([cmd.velocity[0] for cmd in self.vision_commands], label='pred', marker='.') if self.mode == 'vision' else None
            axs[0, 1].plot([cmd.velocity[0] for cmd in self.expert_commands], label='cmd') if self.expert_command is not None else None
            axs[0, 1].plot([vel[0] for vel in self.spline_vels], label='spline vel') if spline_vels is not None else None
            axs[0, 1].plot([vel[0] for vel in self.state_vels], label='state vel')
            axs[0, 1].set_ylabel(f"x vel")
            axs[0, 1].legend()
            axs[0, 1].grid()

            axs[1, 1].plot([cmd.velocity[1] for cmd in self.vision_commands], label='pred', marker='.') if self.mode == 'vision' else None
            axs[1, 1].plot([cmd.velocity[1] for cmd in self.expert_commands], label='cmd') if self.expert_command is not None else None
            axs[1, 1].plot([vel[1] for vel in self.spline_vels], label='spline vel') if spline_vels is not None else None
            axs[1, 1].plot([vel[1] for vel in self.state_vels], label='state vel')
            axs[1, 1].set_ylabel(f"y vel")
            axs[1, 1].legend()
            axs[1, 1].grid()

            axs[2, 1].plot([cmd.velocity[2] for cmd in self.vision_commands], label='pred', marker='.') if self.mode == 'vision' else None
            axs[2, 1].plot([cmd.velocity[2] for cmd in self.expert_commands], label='cmd') if self.expert_command is not None else None
            axs[2, 1].plot([vel[2] for vel in self.spline_vels], label='spline vel') if spline_vels is not None else None
            axs[2, 1].plot([vel[2] for vel in self.state_vels], label='state vel')
            axs[2, 1].set_ylabel(f"z vel")
            axs[2, 1].legend()
            axs[2, 1].grid()

            fig.savefig(f"{self.folder}/cmd_plot.png")

            self.logger.debug(f'Saving plotted commands figure')

            # clear and delete fig
            plt.clf()
            plt.close(fig)

            self.logger.debug(f'Closed figure')

            self.plotted_commands = True

        # # save collected evims
        # if self.state is not None and self.state.pos[0] > self.data_collection_xrange[1] and self.save_events and not self.saved_events:
        #     self.logger.debug(f'Saving evims as npy file')
        #     np.save(f"{self.folder}/evims.npy", self.evims)
        #     self.logger.debug(f'Saving evims to {self.folder}/evims.npy done')
        #     self.saved_events = True

        # # save collected im_dbg2s
        # if self.state is not None and self.state.pos[0] > self.data_collection_xrange[1] and self.save_im_dbg2 and not self.saved_im_dbg2:
        #     self.logger.debug(f'Saving im_dbg2s as npy file')
        #     np.save(f"{self.folder}/im_dbg2s.npy", self.im_dbg2s)
        #     self.logger.debug(f'Saving im_dbg2s to {self.folder}/im_dbg2s.npy done')
        #     self.saved_im_dbg2 = True

    #### END OBSERVATION_CALLBACK

    def log_data(self, log_expert=False):

        # get the current time stamp
        # NOTE use image timestamp since this is important for calculating events
        # and we are using approximate time syncing
        timestamp = np.round(self.im_t, 3)

        data_entry = [
                        timestamp,
                        self.desiredVel,
                        self.state.att[0],
                        self.state.att[1],
                        self.state.att[2],
                        self.state.att[3],
                        self.state.pos[0],
                        self.state.pos[1],
                        self.state.pos[2],
                        self.state.vel[0],
                        self.state.vel[1],
                        self.state.vel[2],
                        self.command.velocity[0] if not log_expert else self.expert_command.velocity[0],
                        self.command.velocity[1] if not log_expert else self.expert_command.velocity[1],
                        self.command.velocity[2] if not log_expert else self.expert_command.velocity[2],
                        self.current_cmd_controller.collective_thrust,
                        self.current_cmd_controller.bodyrates.x,
                        self.current_cmd_controller.bodyrates.y,
                        self.current_cmd_controller.bodyrates.z,
                        self.margin,
                        self.col,
                        ]

        new_row = pd.DataFrame([data_entry], columns=self.data_buffer.columns)
        self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)

        # append data to csv file every data_buffer_maxlength entries
        if len(self.data_buffer) >= self.data_buffer_maxlength:
            self.data_buffer.to_csv(opj(self.folder, 'data.csv'), mode='a', header=not self.first_data_write, index=True)
            self.data_buffer = pd.DataFrame(self.data_format)
            self.first_data_write = True

        # write images every log call
        cv2.imwrite(f"{self.folder}/{timestamp:.3f}_im.png", (self.im*255).astype(np.uint8))
        cv2.imwrite(f"{self.folder}/{timestamp:.3f}_depth.png", (self.depth*255).astype(np.uint8))

        # if self.save_events and not self.saved_events:
        #     self.evims.append(self.events)

        # if self.save_im_dbg2 and not self.saved_im_dbg2:
        #     self.im_dbg2s.append(self.im_dbg2)

    def fix_corrupted_depth(self, depth_image, neighbors=5):
        corrupted_indices = np.where(depth_image == 0.0)
        if len(corrupted_indices) == 0:
            return depth_image
        
        # Iterate through each corrupted pixel
        for i in range(len(corrupted_indices[0])):
            row, col = corrupted_indices[0][i], corrupted_indices[1][i]

            # Extract the neighborhood around the corrupted pixel
            neighborhood = depth_image[max(0, row - neighbors):min(depth_image.shape[0], row + neighbors + 1),
                                    max(0, col - neighbors):min(depth_image.shape[1], col + neighbors + 1)]

            # Exclude the corrupted pixel itself (center of the neighborhood)
            neighborhood = neighborhood[neighborhood != 0.0]

            # Interpolate the corrupted pixel value as the mean of its neighbors
            interpolated_value = np.mean(neighborhood)

            # Assign the interpolated value to the corrupted pixel
            depth_image[row, col] = interpolated_value

        return depth_image

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)
        try:
            self.col = self.if_collide(self.obs_msg.obstacles[0])
        except:
            self.col = 0

    def im_callback(self, im_msg):

        # legacy
        # if self.image_w is None or self.image_h is None:
            # take these values from the config file instead
            # self.image_w = im_msg.width
            # self.image_h = im_msg.height

        try:
            im = self.cv_bridge.imgmsg_to_cv2(im_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("[IM_CALLBACK] CvBridge Error: {0}".format(e))
            return -1
        
        self.im_ctr += 1
        im_t = im_msg.header.stamp.to_nsec() / 1e9 # float with 9 digits past decimal
        self.prev_t = np.round(self.im_t, 3)
        self.im_t = np.round(im_t,3)
        # for rgb images, convert to normalized single channel,
        # preferably in the same way as Vid2E
        if len(im.shape) == 3 or im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im.astype(np.float32)/255.0
        else:
            im = im.astype(np.float32)
        # save image
        self.prev_im = self.im
        self.im = im

        if self.do_events:
            # compute event batch
            self.compute_events(gimbal=False)

        return 0

    def depth_callback(self, depth_msg):

        if self.image_w is None or self.image_h is None:
            self.image_w = depth_msg.width
            self.image_h = depth_msg.height
        try:
            im = self.cv_bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except CvBridgeError as e:
            rospy.logerr("[DEPTH_CALLBACK] CvBridge Error: {0}".format(e))
            return -1
        
        self.depth_t = depth_msg.header.stamp.to_nsec()/1e9
        im = np.clip(im / self.depth_im_threshold, 0, 1)

        self.depth = self.fix_corrupted_depth(im)


        
        # legacy
        # # compute gimbaled images and save
        # q = np.array([self.state.att[0], self.state.att[1], self.state.att[2], self.state.att[3]])
        # self.depth_gimbal, self.pts = self.gimbal.do_gimbal(self.depth, q, self.gimbal_w, self.gimbal_h, do_clip=True)

        return 0

    def obstacle_callback(self, obs_data):
        self.obs_msg = obs_data
        obs = obs_data.obstacles[0]
        # if self.is_trees:
        #     dist = np.linalg.norm(np.array([obs.position.x, obs.position.y]))
        #     # print(f'[EVALUATOR_NODE] obst x, y = [{obs.position.x:.2f}, {obs.position.y:.2f}]')
        # else:
        dist = np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z]))
        # margin is distance to object center minus object radius minus drone radius (estimated)
        self.margin = dist - obs.scale

    def if_collide(self, obs):
        """
        Borrowed and modified from evaluation_node
        """

        if self.is_trees:
            dist = np.linalg.norm(np.array([obs.position.x, obs.position.y]))
        else:
            dist = np.linalg.norm(np.array([obs.position.x, obs.position.y, obs.position.z]))
        # margin is distance to object center minus object radius minus drone radius (estimated)
        margin = dist - obs.scale
        # Ground hit condition
        if margin < 0 or self.state.pos[2] <= 0.1:
            return 1
        else:
            return 0

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                return
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        self.logger.info("Start publishing commands!")
        self.publish_commands = True

    def create_obstacle(self):
        # Create an obstacle with specified position and scale
        obs = Obstacle()
        obs.position = Vector3(1000, 1000, 1000)
        obs.scale = 0.5
        return obs

    def create_obstacle_array(self):
        # Create an ObstacleArray message
        obs_array = ObstacleArray()
        obs_array.header = Header()
        obs_array.header.stamp = rospy.Time.now()
        obs_array.t = rospy.get_time()  # Current time as float64
        obs_array.num = 10  # Number of obstacles

        # Create 10 obstacles and add to the obstacle array
        for _ in range(10):
            obs = self.create_obstacle()
            obs_array.obstacles.append(obs)

        return obs_array


@hydra.main(config_path=opj(base_path2, "configs"), config_name="sim_config",version_base="1.3")
def main(cfg: DictConfig):
    cfg.exp_name = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    agile_pilot_node = AgilePilotNode(cfg)
    rospy.spin()

if __name__ == "__main__":
    main()








