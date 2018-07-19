import os
import re

#Function for adding .flac after each file number to produce the full audio file name
def cleanlist(mylist, regex, substitution):
    tmp_list = mylist
    cleaned_list = [re.sub(regex, substitution, line) for line in tmp_list]
    return cleaned_list

def function(rootdir):
 
    path = rootdir
    audio_file_name = []

    # This for loop allows all sub directories and files to be searched
    for (path, subdirs, files) in os.walk(path):
        files = [f for f in os.listdir(path) if f.endswith('.txt')] # Specify here the format of files you hope to search from (ex: ".txt", ".flac", or ".log")
        files.sort() # file is sorted list
        files = [os.path.join(path, name) for name in files] # Joins the path and the name, so the files can be opened and scanned by the open() function
       
        # The following for loop searches all files with the selected format

        for filename in files:
                
                #Try and except statement in case the text files are encoded differently
                try:
                    with open(filename, 'r', encoding = 'utf-8') as f:
                        f = f.readlines()

                except:
                    with open(filename, 'r') as f:
                        f = f.readlines()                

                # print('Finished parsing... ' + str(datetime.datetime.now())) # For timing the function

                for line in f:
                    #0strip out \x00 from read content, in case it's encoded differently
                    line = line.replace('\x00', '')
                    #return the audio filenames to a list 
                    audio_file_names = re.findall('\d+.\d+.\d+.+', line)
                    #Add ".flac" to the end of each file name
                    audio_file_names = cleanlist(audio_file_names, r'^([\d-]+)', r'\1.flac')
                    #Print the file name, and the corresponding speech text  
                    print(audio_file_names)

                    


