fp = '/home/zachuhlmann/projects/basin_masks/color_ramp_diagnose2.qml'
fp_out = '/home/zachuhlmann/projects/basin_masks/color_ramp_diagnose.qml'

lst_old = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
lst_new = col_ramp
str_nums = [str(num) for num in lst_old]
str_nums2 = [str(num) for num in lst_new]

zip_lst = list(zip(str_nums, str_nums2))

with open(fp) as infile, open(fp_out, 'w') as outfile:
    for line in infile:
        for i in range(len(zip_lst)):
            line = line.replace(zip_lst[i][0], zip_lst[i][1])
            print(zip_lst[i][0], zip_lst[i][1])
        outfile.write(line)
