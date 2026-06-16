import os

import numpy as np


if __name__ == "__main__":


    logs_dir = None #TODO add path
    versions = ["version_0","version_1","version_2","version_3","version_4"]
    result_file_ending = "ma_results.txt"
    all_results = dict()


    for version in versions:
        directories = os.listdir(os.path.join(logs_dir,version,'checkpoints'))
        directories=[d for d in os.listdir(os.path.join(logs_dir,version,'checkpoints')) if os.path.isdir(os.path.join(os.path.join(logs_dir,version,'checkpoints',d)))]
        epochs = []
        for element in directories:
            epochs.append(int(element.split('=')[1].split('-')[0]))

        last_directory = directories[np.array(epochs).argmax()]

        with open(os.path.join(logs_dir,version,'checkpoints',last_directory,result_file_ending), 'r') as f:
            for line in f:
                tokens = line.split(":")
                if all_results.get(tokens[0]) is None:
                    all_results[tokens[0]] = []
                all_results[tokens[0]].append(float(tokens[1]))



    results_file = open(os.path.join(logs_dir,"accumulated_"+result_file_ending),"w")
    results_file.write(" Combined results for versions: "+'\n' )
    for version in versions:
        results_file.write(version + ", ")
    results_file.write('\n')

    for key in all_results.keys():
        print(key," : mean ", np.round(np.array(all_results[key]).mean(),decimals=4)," std ",np.round(np.array(all_results[key]).std(),decimals=4))
        results_file.write(key + " : mean "+ str(np.round(np.array(all_results[key]).mean(),decimals=4))+" std "+str(np.round(np.array(all_results[key]).std(),decimals=4)))
        results_file.write(" \n")
    results_file.close()
