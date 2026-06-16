from Evaluation import  *



def delete_too_small_fronts(front, front_length_threshold):
    # delete too short fronts
    labeled_front, num_cluster = skimage.measure.label(front, connectivity=2, return_num=True)
    if num_cluster == 0:
        return front * 255

    #cuda out of memory. Ooopsies doopsie
    cluster_frequencies = (torch.from_numpy(labeled_front).to('cuda')).unique(return_counts=True)[1]
    idx_of_kept_cluster = torch.argwhere(torch.where(cluster_frequencies>=front_length_threshold,cluster_frequencies,0))
    front = np.array((torch.where( torch.isin(torch.from_numpy(labeled_front).to('cuda'),idx_of_kept_cluster.flatten()[1:]),1,0)*255).to('cpu'))

    return front


def remove_catchment(image, name,ice_mask_directory):
    name = ('_').join(name.split('_')[:-3])

    path = os.path.join(ice_masks_directory,name+".png")
    ice_mask = np.array(Image.open(path).convert('L'))
    image[ice_mask == 0] = 0
    return image

def load_fronts_known(front_directory):
    front_dictionary = dict()
    for file in os.listdir(front_directory):
        img = Image.open(os.path.join(front_directory,file)).convert('L')
        name = file[:-4]
        front_dictionary[name] = img

    return front_dictionary

if __name__ == "__main__":
    #input directory of the segmented fronts

    logs_dir = None #TODO LogsDir
    versions = [] #TODO
    ice_masks_directory = None# TODO icemask dir
    directory_of_multiannotator_fronts = None# TODO

    for version in versions:
        directories = os.listdir(os.path.join(logs_dir,version,'checkpoints'))
        directories=[d for d in os.listdir(os.path.join(logs_dir,version,'checkpoints')) if os.path.isdir(os.path.join(os.path.join(logs_dir,version,'checkpoints',d)))]
        epochs = []
        for element in directories:
            epochs.append(int(element.split('=')[1].split('-')[0]))

        last_directory = directories[np.array(epochs).argmax()]
        directory_of_source_fronts = os.path.join(logs_dir,version,'checkpoints',last_directory,'output_images','complete_postprocessed_images')

        results_txt_path = os.path.join((os.sep).join(directory_of_source_fronts.split(os.sep)[:-2]),"ma_results.txt")
        with open(results_txt_path,'w') as f:
            a=1


        meter_threshold = 750
        log_list = []

        post_processed_dictionary = load_fronts_known(front_directory=directory_of_source_fronts)
        names = list(post_processed_dictionary.keys())

        #clean up for multi annotator
        for name in names:
            a=1
            post_processed_dictionary[name] = Image.fromarray(remove_catchment(np.array(post_processed_dictionary[name]),name,ice_masks_directory),mode='L')
            resolution = int(name.split('_')[-3])
            pixel_threshold = meter_threshold / resolution
            post_processed_dictionary[name] = Image.fromarray(delete_too_small_fronts(np.array(post_processed_dictionary[name]), pixel_threshold).astype(np.uint8),mode='L')

        """"""
        #Multi Annotator Labels
        log_list = []
        front_dictionary = load_fronts(front_directory=directory_of_multiannotator_fronts)
        front_delineation_metric(names, post_processed_dictionary, front_dictionary, log_list,results_txt_path=results_txt_path)


