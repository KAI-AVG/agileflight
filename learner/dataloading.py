# %%
import shutil
import glob, os, time
from os.path import join as opj
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import cv2
import re
import h5py
import sys
base_path=os.environ.get("project_path")
sys.path.append(opj(base_path,"event_denoise"))
from collections import OrderedDict
from typing import Tuple,List
from torch.utils.data import Dataset
from scipy.signal import convolve2d
import matplotlib
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig
#matplotlib.use('TkAgg')

# %%
def find_unmatched_indices(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    unmatched_indices_1 = [i for i, val in enumerate(list1) if val not in set2]
    unmatched_indices_2 = [i for i, val in enumerate(list2) if val not in set1]

    return unmatched_indices_1, unmatched_indices_2

def load_im_as_gray(f):
    return cv2.cvtColor(cv2.imread(f, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)[...,:3].mean(-1).astype(np.uint8)

# %%
class Data:
    def __init__(self,data_path:str,dataset_name:List[str],short:List[int],dvs_num:int=2,val_split:float=0.2,split_method='train-val',event_batch:int=2,denoising:str=None,rescale_depth:float=1,keep_collisions:bool=False,resize_input:Tuple[int,int]=[60,90]):
        # 配置路径
        self.data_path = data_path
        self.dvs_num = dvs_num
        self.val_split = val_split
        if isinstance(dataset_name,str):
            dataset_name = [f'{dataset_name}']
        self.datasets_name = dataset_name
        self.rescale_depth = rescale_depth
        assert split_method in ['train-val','val-train']
        self.split_method = split_method
        self.event_batch = event_batch
        self.keep_collisions = keep_collisions
        self.resize_input = resize_input
        self.denoising = denoising 
        if self.denoising:
            assert self.denoising in ['connective','mlpfilter','stcf']
        #self._dataset_path = [opj(self.data_path,dataset_name) for dataset_name in self.datasets_name]
        self.short = short
        assert any(i >= 0 for i in self.short)
        # 构造数据集字典
        self._data_template=['traj_meta', 'traj_ims', 'traj_depths', 'traj_vels', 'traj_evs', 'desired_vels',"traj_lengths","traj_evs_img","obs_dis","traj_state"]
        self._dataset= OrderedDict((dataset, OrderedDict((data, None) for data in self._data_template)) for dataset in self.datasets_name)
    def dataloading(self):
        self._load_dataset()
        self.train_data = OrderedDict((data, None) for data in self._data_template)
        self.val_data = OrderedDict((data, None) for data in self._data_template)
        all_data = OrderedDict((data, []) for data in self._data_template)
        all_data.update((key, [self._dataset[_dataset][key] for _dataset in self._dataset.keys()]) for key in ["traj_ims","traj_depths","traj_vels","desired_vels","traj_meta","traj_lengths","traj_evs_img","obs_dis","traj_state"])
        all_data.update((key, torch.cat(all_data[key])) for key in ["traj_ims","traj_depths","traj_vels","desired_vels","traj_meta","traj_evs_img","traj_lengths","obs_dis","traj_state"])
        all_data["traj_evs"].extend(item for _dataset in self._dataset.keys() for item in self._dataset[_dataset]["traj_evs"])
        if self.split_method == 'train-val':
            num_train_trajs = int((1.-self.val_split) * all_data["traj_lengths"].shape[0])
            train_trajs_st = 0
            train_trajs_end = num_train_trajs
            val_trajs_st = num_train_trajs
            val_trajs_end = all_data["traj_lengths"].shape[0]
            train_idx_st = 0
            train_idx_end = sum(all_data["traj_lengths"][:num_train_trajs])
            val_idx_st = train_idx_end
            val_idx_end = sum(all_data["traj_lengths"])
        elif self.split_method == 'val-train':
            num_val_trajs = int(self.val_split * all_data["traj_lengths"].shape[0])
            val_trajs_st = 0
            val_trajs_end = num_val_trajs
            train_trajs_st = num_val_trajs
            train_trajs_end = all_data["traj_lengths"].shape[0]
            val_idx_st = 0
            val_idx_end = sum(all_data["traj_lengths"][:num_train_trajs])
            train_idx_st = val_idx_end
            train_idx_end = sum(all_data["traj_lengths"])
        train_range=slice(train_idx_st,train_idx_end)
        val_range=slice(val_idx_st,val_idx_end)
        self.train_data.update({key: all_data[key][train_range] for key in ["traj_meta", "traj_ims", "traj_vels","traj_depths","desired_vels","traj_evs","traj_evs_img","obs_dis","traj_state"]})
        self.val_data.update({key: all_data[key][val_range] for key in ["traj_meta", "traj_ims", "traj_vels","traj_depths","desired_vels","traj_evs","traj_evs_img","obs_dis","traj_state"]})
        train_range=slice(train_trajs_st,train_trajs_end)
        val_range=slice(val_trajs_st,val_trajs_end)
        self.train_data.update({key:all_data[key][train_range] for key in ["traj_lengths"]})
        self.val_data.update({key:all_data[key][val_range] for key in ["traj_lengths"]})
    

    def _load_dataset(self):
        for i in range(len(self.datasets_name)):
            _dataset_name = self.datasets_name[i]
            _short = self.short[i]
            _data_path=opj(base_path,self.data_path,_dataset_name)
            self._dataset[_dataset_name] = self._load_single_dataset(data_path=_data_path,short=_short)
    def _load_single_dataset(self,data_path:str,short:int=0):
        single_data = OrderedDict((data, []) for data in self._data_template)
        if os.path.exists(f'{data_path}.h5') and os.path.exists(f'{data_path}.pth'):
            single_data=self.load(input_path=data_path)
        else:
            neg_thresh = 0.1
            pos_thresh = 0.1
            kept_folder_ids_shuffled=[]
            traj_folders =  sorted(folder for folder in glob.glob(opj(data_path, '*')) if os.path.isdir(folder))
            traj_folders_keys = [i for i in range(len(traj_folders))]
            if short > 0:
                assert short <= len(traj_folders), f"short={short} is greater than the number of folders={len(traj_folders)}"
                traj_folders= traj_folders[:short]
                traj_folders_keys = traj_folders_keys[:short]
            num_collision_trajs = 0
            k = 0
            for traj_i, traj_folder in enumerate(traj_folders):   
                #self.logger.info(f'Loading folder {traj_folder}')
                csv_file = 'data.csv'
                try:
                    traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
                except:
                    # Open the CSV file and read its contents
                    with open(opj(traj_folder, csv_file), 'r') as file:
                        # Read the CSV file line by line
                        lines = file.readlines()
                        
                        traj_meta = []
                        # Iterate through the lines and exclude the line with the wrong number of columns
                        for _, line in enumerate(lines[1:]):
                            num_columns = len(line.strip().split(','))
                            if num_columns != 21:
                                continue

                            traj_meta.append([float(x) for x in line.strip().split(',')])  # Split the line and convert values to float
                        traj_meta = np.array(traj_meta, dtype=np.float64)
                # check for nan in metadata
                if np.isnan(traj_meta).any():
                    self.logger.info(f'Deleting folder {os.path.basename(traj_folder)}')
                    shutil.rmtree(traj_folder)
                    continue
                # check for collisions in trajectory
                if traj_meta[:,-1].sum() > 0:
                    num_collision_trajs += 1
                    #self.logger.debug(f"{traj_meta[:,-1].sum()} collisions in {os.path.basename(traj_folder)}, {num_collision_trajs}th so far, {'skipping!' if not keep_collisions else 'keeping!'}")
                    if not self.keep_collisions:
                        continue
                    # find image and depths filenames
                    # newer datasets have both depth images *_depth.png and gray images *_im.png
                depth_files = sorted(glob.glob(opj(traj_folder, '*_depth.png')))
                im_files = sorted(glob.glob(opj(traj_folder, '*_im.png')))
                # check for empty folder
                if len(im_files) == 0 or len(depth_files)==0:
                    shutil.rmtree(traj_folder)
                    continue
                traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in im_files], dtype=np.float32) / 255.0
                traj_depths = np.asarray([cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE) for depth_file in depth_files], dtype=np.float32) / 255.0

                traj_ims_ts = []
                flag=False
                for im_i in range(traj_ims.shape[0]):
                    # im_timestamp = os.path.basename(im_files[im_i])[:-4]
                    match = re.search(r'(\d+(\.\d+)?)', os.path.basename(im_files[im_i]))
                    # Check if there is exactly one match
                    # match groups contains "the whole match" and then each subcomponent, so index 1 is the first relevent one
                    if match is not None and len(match.groups()) == 2:
                        im_timestamp = match.group(1)
                        # self.logger.info(f'[DATALOADER] Found timestamp {im_timestamp} in image filename {im_files[im_i]}')
                        # self.logger.info('THIS WAS A DEBUG MSG, QUITTING NOW!')
                        # exit()
                    else:
                        # if do_clean_dataset:
                        #     logger(f'[DATALOADER] Deleting folder {os.path.basename(traj_folder)}')
                        #     os.system(f'rm -r {traj_folder}')
                        shutil.rmtree(traj_folder)
                        flag =True
                    traj_ims_ts.append(float(im_timestamp))
                if flag:
                    continue
                # check for duplicates in metadata timestamps
                # since collisions force logging, there may be duplicate timestamps
                _, unique_indices, counts = np.unique(traj_meta[:, 1], return_index=True, return_counts=True)
                duplicate_indices = unique_indices[counts > 1]
                traj_meta = np.delete(traj_meta, duplicate_indices, axis=0)
                # find indices of mismatched timestamps
                #st_match = time.time()
                unmatched_indices_1, unmatched_indices_2 = find_unmatched_indices(traj_ims_ts, list(traj_meta[:, 1]))
                if len(unmatched_indices_1) > 0 or len(unmatched_indices_2) > 0:
                    traj_ims = np.delete(traj_ims, unmatched_indices_1, axis=0)
                    for im_idx in unmatched_indices_1:
                        print(f'Deleting image {im_files[im_idx]}')
                        os.system(f'rm {im_files[im_idx]}')
                    if len(depth_files) > 0:
                        traj_depths = np.delete(traj_depths, unmatched_indices_1, axis=0)
                        for depth_idx in unmatched_indices_1:
                            print(f'Deleting depth {depth_files[depth_idx]}')
                            os.system(f'rm {depth_files[depth_idx]}')
                    traj_meta = np.delete(traj_meta, unmatched_indices_2, axis=0)
                    #self.logger.debug(f'\tTime to find and delete unmatched indices: {time.time()-st_match:.3f}s')
                if self.resize_input:
                    traj_ims = self._resize_input(traj_ims,resize_shape=self.resize_input)
                    traj_depths = self._resize_input(traj_depths,resize_shape=self.resize_input)
                if self.rescale_depth:
                    traj_depths = np.clip(traj_depths/self.rescale_depth,0,1)
                unmatched_ids_ims=unmatched_indices_1

                # convert metadata to np.float32 after matching timetamps and ims/depths is done
                # 0-start all timestamps in column 1
                traj_meta[:, 1] -= traj_meta[0, 1]
                # convert to np.float32
                traj_meta = np.array(traj_meta, dtype=np.float32)
                timestamps = traj_meta[:,1]
                evframes = self._convert_event(timestamps=timestamps,images=traj_ims,neg_thresh=neg_thresh,pos_thresh=pos_thresh,batch_size=self.event_batch,denoising=self.denoising) 
                # if motion_blur:
                #     speeds = traj_meta[1:, 2]
                #     # depth_image = traj_depths[1]
                #     # depth_image = (depth_image * 255).astype(np.uint8)
                #     # depth_image = traj_depths[0]
                #     # # 显示图片
                #     # plt.imshow(depth_image, cmap='gray')
                #     # plt.axis('off')  # 关闭坐标轴
                #     # plt.show()
                #     traj_depths[1:] = np.array([self.add_motion_blur(depth, speed, motion_blur) for depth, speed in zip(traj_depths[1:], speeds)])
                #     # depth_image = traj_depths[1]
                #     # depth_image = (depth_image * 255).astype(np.uint8)
                #     # plt.imshow(depth_image, cmap='gray')
                #     # plt.axis('off')  # 关闭坐标轴
                #     # plt.show()
                # if k<10:
                #     save_depth_folder = opj(base_path, 'data/saved_depths')
                #     save_img_folder = opj(base_path,"data/saved_images")
                #     os.makedirs(save_depth_folder, exist_ok=True)
                #     os.makedirs(save_img_folder,exist_ok=True)
                #     for i in range(traj_depths.shape[0]):
                #         if k >= 10:
                #             break
                #         save_depth_path = opj(save_depth_folder, f'image_{k}.png')
                #         save_img_path = opj(save_img_folder,f'image_{k}.png')
                #         k += 1
                #         # 保存图片
                #         plt.imsave(save_depth_path, traj_depths[i], cmap='gray')
                #         plt.imsave(save_img_path, traj_ims[i],cmap='gray')
                single_data["traj_ims"].append(traj_ims[1:])
                single_data["traj_meta"].append(traj_meta[1:])
                single_data["traj_depths"].append(traj_depths[1:])
                single_data["desired_vels"].extend(traj_meta[1:, 2])
                single_data["traj_evs"].extend(evframes)
                kept_folder_ids_shuffled.append(traj_i)
            print(len(single_data["traj_ims"]))
            single_data["traj_lengths"] = torch.tensor([len(traj_ims) for traj_ims in single_data["traj_ims"]])
            img_h,img_w = single_data["traj_ims"][0].shape[-2:]
            meta_length = single_data["traj_meta"][0].shape[-1]
            single_data["traj_evs_img"] = torch.from_numpy(self._process_event(events=single_data["traj_evs"],img_size=(img_h,img_w),S=5,denoising=self.denoising,dvs_num=self.dvs_num))
            ## to tensor
            keys = ["traj_ims","traj_depths"]
            single_data.update({key: torch.cat([torch.from_numpy(arr) for arr in single_data[key]]).reshape(-1,img_h,img_w) for key in keys})
            single_data["traj_meta"] = torch.cat([torch.from_numpy(arr) for arr in single_data["traj_meta"]]).reshape(-1,meta_length)
            single_data['traj_vels']=single_data["traj_meta"][:,13:16]/single_data["traj_meta"][:,2].unsqueeze(1)
            single_data["obs_dis"] = single_data["traj_meta"][:,20]
            single_data["traj_state"] = single_data["traj_meta"][:,3:13]
            single_data["desired_vels"] = torch.tensor(single_data["desired_vels"])
            single_data["traj_evs"] = [torch.from_numpy(evs) for evs in single_data["traj_evs"]]
            save_path = data_path
            # if motion_blur:
            #     save_path += f"_motion{motion_blur}"
            if self.denoising:
                save_path += f"_denoising{self.denoising}"
            if short:
                save_path += f"_short{short}"
            # if self.dvs_num ==2 :
            #     save_path += f"_dvs_num{self.dvs_num}"
            self.save(output_path=save_path,single_dataset=single_data)
        return single_data
    
    # def _process_event(self,event_windows:List[object])->np.array:
    #     for event in event_windows:
    def _process_event(self,events:List[np.array],img_size:Tuple[int,int],S:int,denoising:str=None,dvs_num:int=4)->np.array:
        event_img=[]
        k = 0
        for event_array in events:
            if dvs_num == 4:
                evframes = np.zeros((4,*img_size))
                if event_array.shape[0]:
                    t_norm = event_array[-1,2]-event_array[0,2]
                    if t_norm != 0:
                        event_array[:, 2] = event_array[:, 2] - event_array[0, 2]
                        event_array[:, 2] = event_array[:, 2] / t_norm * S
                    else:
                        event_array[:, 2] = 0
                    pos_indices = event_array[:, 3] > 0
                    neg_indices = ~pos_indices
                    # 处理正负极性事件
                    evframes[0, event_array[pos_indices, 1].astype(int), event_array[pos_indices, 0].astype(int)] += 1
                    evframes[2, event_array[pos_indices, 1].astype(int), event_array[pos_indices, 0].astype(int)] = event_array[pos_indices, 2]
                    evframes[1, event_array[neg_indices, 1].astype(int), event_array[neg_indices, 0].astype(int)] += 1
                    evframes[3, event_array[neg_indices, 1].astype(int), event_array[neg_indices, 0].astype(int)] = event_array[neg_indices, 2] 
                    if denoising == 'connective':
                        evframes = self.connetive_denoising(evframes=evframes,denoising=denoising)
            elif dvs_num == 2:
                evframes = np.zeros((2,*img_size))
                if event_array.shape[0]:
                    t_norm = event_array[-1,2]-event_array[0,2]
                    if t_norm != 0:
                        event_array[:, 2] = event_array[:, 2] - event_array[0, 2]
                        event_array[:, 2] = event_array[:, 2] / t_norm * S
                    else:
                        event_array[:, 2] = 0
                    evframes[0, event_array[:, 1].astype(int), event_array[:, 0].astype(int)] += 1
                    evframes[1, event_array[:, 1].astype(int), event_array[:, 0].astype(int)] = event_array[:, 2]
                    if denoising=='connect':
                        evframes = self.connetive_denoising(evframes=evframes,denoising=4)
            event_img.append(evframes)
            if  k<10:
                self.preview_event(evframes=evframes,img_size=img_size,k=k,dvs_num=dvs_num)
                k += 1
        return np.array(event_img)
    
    def connetive_denoising(self,evframes:np.array,denoising:int)->np.array:
        """
        对事件帧进行连通性去噪处理。

        :param evframes: 输入的事件帧数组，包含不同通道的事件信息。
        :param denoising: 去噪阈值，面积小于该阈值的连通区域将被移除。
        :return evframes: 经过去噪处理后的事件帧数组。
        """
        pos_img = (evframes[0]*10).astype(np.int8)
        retval,labels,stats,centroids = cv2.connectedComponentsWithStats(pos_img,connectivity=8)
        if retval >1:
            for i in range(1,retval):
                if stats[i,cv2.CC_STAT_AREA] < denoising:
                    labels[labels==i] =0
        for channel in [0,1]:
            evframes[channel][labels == 0] = 0
        # neg_img = (evframes[1]*10).astype(np.int8)
        # retval,labels,stats,centroids = cv2.connectedComponentsWithStats(neg_img,connectivity=8)
        # if retval >1:
        #     for i in range(1,retval):
        #         if stats[i,cv2.CC_STAT_AREA] < denoising:
        #             labels[labels==i] =0
        # for channel in [1,3]:
        #     evframes[channel][labels == 0] = 0
        return evframes

    
    def preview_event(self,evframes:np.array,img_size:Tuple[int],k:int,dvs_num:int=2):
        fig, axes = plt.subplots(1, dvs_num, figsize=img_size)
        for i in range(dvs_num):
            axes[i].imshow(evframes[i], cmap='gray')
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')
        #plt.show()
        save_folder = opj(base_path,"data/event_image")
        os.makedirs(save_folder,exist_ok=True)
        save_path = opj(save_folder,f"image_{k}")
        plt.savefig(save_path)
        plt.close(fig)
        
    def add_motion_blur(self,depth_image, speed, motion_size)->np.array:
        kernel_size = int(speed * motion_size) 
        kernel = np.diag(np.ones(kernel_size))
        kernel = kernel / kernel_size
        # 使用numpy进行卷积操作
        blurred_image = convolve2d(depth_image, kernel, mode='same')
        blurred_image = np.clip(blurred_image, 0, 1).astype(depth_image.dtype)
        return blurred_image

        
    def _resize_input(self,img_array:np.array,resize_shape:Tuple[int,int])->np.array:
        img_array=torch.from_numpy(img_array)
        img_array=torch.nn.functional.interpolate(img_array.unsqueeze(1), size=tuple(resize_shape), mode='bilinear', align_corners=False).squeeze()
        return img_array.numpy()
    

    
    def save(self,output_path:str,single_dataset:OrderedDict):
        h5_path=f'{output_path}.h5'
        with h5py.File(h5_path, 'w') as hf:
            for key, value in single_dataset.items():
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                if isinstance(value,np.ndarray):
                    hf.create_dataset(key, data=value)
            hf.close()
        torch.save(single_dataset["traj_evs"],f"{output_path}.pth")
        

    def load(self, input_path: str) -> OrderedDict:
        single_data = OrderedDict()
        with h5py.File(f'{input_path}.h5', 'r') as hf:
            for key in hf.keys():
                data = hf[key][()]
                data = torch.from_numpy(data)
                single_data[key] = data
        single_data["traj_evs"] = torch.load(f'{input_path}.pth',weights_only=False)
        return single_data

        #self.logger.info(f'Saved dataset to {h5_path}')
    # convert image to event
    def _convert_event(self,timestamps:np.array,images:np.array,neg_thresh:int,pos_thresh:int,batch_size:int,denoising:str=None)->np.array:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        import esim_torch
        events = []
        esim = esim_torch.ESIM(contrast_threshold_neg=neg_thresh,
                            contrast_threshold_pos=pos_thresh,
                            refractory_period_ns=.5e6) # note this is in ns

        timestamps_ns = (timestamps * 1e9).astype("int64")
        num_frames = len(timestamps_ns) - 1
        log_images = np.log(images.astype("float32") + 1e-10)


        # generate torch tensors
        device = torch.device("cuda")
        log_images = torch.from_numpy(log_images).to(device)
        timestamps_ns = torch.from_numpy(timestamps_ns).to(device)

        # generate events with GPU support
        traj_events = []
        for i in range(0, num_frames, batch_size):
            with torch.no_grad():
                traj_events.append(esim.forward(log_images[i:i+batch_size], timestamps_ns[i:i+batch_size]))
            for k in traj_events[-1].keys():
                traj_events[-1][k] = traj_events[-1][k].cpu()


        # compile batches into a single dict
        full_batch = {'x': torch.Tensor([]), 'y': torch.Tensor([]), 't': torch.Tensor([]), 'p': torch.Tensor([])}
        for ev_batch in traj_events:
            full_batch['x'] = torch.cat((full_batch['x'], ev_batch['x']))
            full_batch['y'] = torch.cat((full_batch['y'], ev_batch['y']))
            full_batch['t'] = torch.cat((full_batch['t'], ev_batch['t']))
            full_batch['p'] = torch.cat((full_batch['p'], ev_batch['p']))
        
        events = full_batch
        timestamps_ns = timestamps_ns.cpu().numpy()
        log_images = log_images.cpu().numpy()
        
        ts = events['t']
        
        p = events['p']
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # one fewer frames than the number of samples
        windows = []
        for i in range(images.shape[0]-1):

            t_start = timestamps_ns[i] - timestamps_ns[0]
            t_end = timestamps_ns[i+1] - timestamps_ns[0]
            event_indices = torch.bitwise_and((ts >= t_start), (ts < t_end))
            x, y, t, p = torch.stack([events[key][event_indices].unsqueeze(1) for key in ['x', 'y', 't', 'p']], dim=0).unbind(0)
            event_window = torch.cat((x,y,t,p),dim=1)
            sorted_indices = torch.argsort(event_window[:, 2]) 
            event_window = event_window[sorted_indices]
            if event_window.shape[0]>1:
                if denoising == 'mlpfilter':
                    from mlpffilteringEvents import mlpffiltering # type: ignore
                    event_window = event_window.numpy()
                    event_window = event_window[:,[0,1,3,2]]
                    event_window[:,3] = event_window[:,3]/1e6
                    denoise_event = mlpffiltering(event_window,0.5,"ds")
                    event_window[:,3] = event_window[:,3]*1e6
                    denoise_event = denoise_event[:,[0,1,3,2]]
                    event_window = torch.tensor(denoise_event)
                elif denoising == "stcf":
                    from stcfforshang import stcfforshang # type: ignore
                    event_window = event_window.numpy()
                    event_window = event_window[:,[2,0,1,3]]
                    event_window[:,0] = event_window[:,0]/1e3
                    denoise_event = stcfforshang(event_window)
                    event_window[:,0] = event_window[:,0]*1e3
                    denoise_event = denoise_event[:,[1,2,0,3]]
                    event_window = torch.tensor(denoise_event)
            windows.append(event_window.numpy().astype(np.float32))
    
    
        # windows_difflog=[]
        # for im_i in range(images.shape[0]-1):

        #     # approximation of events calculation
        #     difflog = log_images[im_i+1] - log_images[im_i]

        #     # thresholding
        #     events_frame_difflog = np.zeros_like(difflog)

        #     if not np.abs(difflog).max() < max(pos_thresh, neg_thresh):

        #         # quantize difflog by thresholds
        #         pos_idxs = np.where(difflog > 0.0)
        #         events_frame_difflog[pos_idxs] = 1
        #         neg_idxs = np.where(difflog < 0.0)
        #         events_frame_difflog[neg_idxs] = -1
            
        #     non_zero_indices = np.nonzero(events_frame_difflog)
        #     x_coords = non_zero_indices[1]
        #     y_coords = non_zero_indices[0]
        #     p_values = events_frame_difflog[non_zero_indices]
        #     window_difflog = np.column_stack((x_coords, y_coords, p_values))
        #     windows_difflog.append(window_difflog[np.newaxis:])
        
        # np.save(opj(output_path, 'evs_winodws.npy'), windows)
        # np.save(opj(output_path, 'evs_winodws_diff.npy'),windows_difflog)

        return windows

# %%
class CustomDataset(Dataset):
    def __init__(self,dataset:OrderedDict,batch_size:int=10):
        self.count = 0
        self._dataset=[]
        traj_lengths = dataset["traj_lengths"]
        start_ids = torch.cumsum(traj_lengths,dim=0)-traj_lengths
        self._dataset.extend([
            OrderedDict(
                (key, dataset[key][slice(start_ids[i] + j * batch_size, start_ids[i] + (j + 1) * batch_size)])
                for key in ["traj_meta", "traj_ims", "traj_depths", "traj_evs", "traj_vels", "desired_vels","traj_evs_img","obs_dis","traj_state"]
            )
            for i in range(len(traj_lengths))
            for j in range(traj_lengths[i] // batch_size)
        ])
    def __iter__(self):
        return self
    
    def __getitem__(self,idx):
        return self._dataset[idx]
    
    def __len__(self):
        return len(self._dataset)
    
    def __next__(self):
        if self.count < len(self._dataset):
            data = self._dataset[self.count]
            self.count += 1
            return data
        else:
            raise StopIteration



@hydra.main(config_path=opj(base_path,"configs"), config_name="dataloading",version_base="1.3")
def main(cfg: DictConfig):
    dataset = instantiate(cfg.dataset)
    dataset.dataloading()


if __name__ == "__main__":
    main()
# # #%%
# if __name__ == "__main__":
#     data=Data(data_path="data/datasets",dataset_name=["easy_dynamic_17_all"],short=[0])
#     data.dataloading(from_h5=False,motion_blur=0,event_batch=2,denoising=None,dvs_num=2)

# trainset=CustomDataset(data.train_data)
# valset=CustomDataset(data.val_data)
# sample=DistributedSampler(trainset,num_replicas=2,rank=1)
# traindataloader=DataLoader(trainset,sampler=sample)
# for i in traindataloader:
#     a=i
#     print(i)
