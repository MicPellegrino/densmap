import os

folder_name = "Flow/"
file_root = "flow_"
file_range = [1, 330]
offset = 329

if offset > 0 :
    for n in range(file_range[0], file_range[1]) :
        old_file_name = folder_name+file_root+str(file_range[1]-n).zfill(5)+'.dat'
        new_file_name = folder_name+file_root+str(file_range[1]-n+offset).zfill(5)+'.dat'
        os.system("mv "+old_file_name+" "+new_file_name)
        print(old_file_name+"->"+new_file_name)
else :
    for n in range(file_range[0], file_range[1]) :
        old_file_name = folder_name+file_root+str(n).zfill(5)+'.dat'
        new_file_name = folder_name+file_root+str(n+offset).zfill(5)+'.dat'
        os.system("mv "+old_file_name+" "+new_file_name)
        print(old_file_name+"->"+new_file_name)
