import os
import os.path as osp

import argparse

from moviepy import VideoFileClip, clips_array, concatenate_videoclips


def get_video_idx(video):
    print(video.split('_'))
    
    if '_gt' in video:
        video_idx = '_'.join(video.split('_')[:-2])
    else:
        video_idx = '_'.join(video.split('_')[:-1])
    
    return video_idx


def combine_video(target_directory):
    
    video_list = [f for f in os.listdir(target_directory) if '_mesh.mp4' in f]

    combined_videos = []
    
    for video in video_list:
        video_idx = get_video_idx(video)
        
        if video_idx in combined_videos:
            continue
    
        video1 = osp.join(target_directory, f'{video_idx}_mesh.mp4')
        video2 = osp.join(target_directory, f'{video_idx}_gt_mesh.mp4')
        
        animation_clip1 = VideoFileClip(video1)
        animation_clip2 = VideoFileClip(video2)
        
        final_clip = clips_array([[animation_clip1, animation_clip2]])
        output = osp.join(target_directory, f'{video_idx}_mesh_combined.mp4')
        final_clip.write_videofile(output)



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Combine videos')
    parser.add_argument('—dir', type=str, help='Directory containing videos to combine')
    
    args = parser.parse_args()
    
    combine_video(args.dir)