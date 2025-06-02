from PF2nlfff.packages import *
from PF2nlfff.tools import *
from PF2nlfff.nlfff_module import *

# Divide the training set and the test set
product_info_path = '../archive-202203-info.csv'
product_info = pandas.read_csv(product_info_path)

path = '/data3/nlfff/grid3.done/*'
dirs_list = glob.glob(path)

data_set_list = []

for i in range(len(dirs_list)):
    file_list = get_files_with_path(dirs_list[i])
    sample_harpnum_trec_list = []
    for i in range(len(file_list)):
        match = re.search(r'\d+\.\d{8}_\d{6}_\w+', file_list[i])
        if match:
            extracted_part = match.group()
            sample_harpnum_trec = extracted_part
            sample_harpnum_trec_list.append(sample_harpnum_trec)

    sample_product_info = product_info[product_info['harpnum_trec'].isin(sample_harpnum_trec_list)]
    min_grid_xyz = min(sample_product_info["grid_xyz"])
    filtered_data = sample_product_info[sample_product_info['grid_xyz'] == min_grid_xyz]
    file_name_list = pd.DataFrame({'file_name':file_list})
    filtered_harpnum_trec_list = filtered_data["harpnum_trec"].tolist()
    filtered_file_name_list = file_name_list[file_name_list["file_name"].str.contains('|'.join(filtered_harpnum_trec_list))]
    data_set_list.append(filtered_file_name_list["file_name"].tolist())
    
file_size_list = []
for i in range(len(data_set_list)):
    file_size = os.path.getsize(data_set_list[i][0])
    file_size_list.append(file_size)
    
sorted_index_list = sorted(range(len(data_set_list)), key=lambda x: file_size_list[x], reverse=True)
data_set_sorted_by_size = [data_set_list[i] for i in sorted_index_list]
min_file_nums = 10000
file_num_list = []
idx = 0
for i in range(len(data_set_sorted_by_size[10:])):
    if len(data_set_sorted_by_size[10:][i]) <= min_file_nums:
        idx = i+1
        min_file_nums = len(data_set_sorted_by_size[10:][i])
    file_num_list.append(len(data_set_sorted_by_size[10:][i]))
sorted_index_list = sorted(range(len(file_num_list)), key=lambda x: file_num_list[x], reverse=True)
sorted_data_set_list = [data_set_sorted_by_size[10:][i] for i in sorted_index_list]
test_ds = sorted_data_set_list[5:]
training_ds = sorted_data_set_list[:5]
files_training = [item for sublist in training_ds for item in (sublist if isinstance(sublist, list) else [sublist])]
files_test     = [item for sublist in test_ds for item in (sublist if isinstance(sublist, list) else [sublist])]
print(f"Numbers of training samples: {len(files_training)}")
print(f"Numbers of test samples    : {len(files_test)}")

# Get the potential field for all the training set data
cnt = 0
total = len(files_training)
    
for file in file_list:
    cnt+=1
    b_cube = read_bcube(file)
    save_name = file[:-8]+'PF_data.npy'
    progress = "#" * int(100 * cnt / total)
    percentage1 = 100 * (cnt) / total
    fraction1 = "{}/{}".format(cnt, total)
    print("\rProgressing 1:[{0:100s}] {1:.1f}% ({2})".format(progress, percentage, fraction))
    if os.path.exists(save_name):
        is_calculate_PF = False
        print('\r[file:] %s exists' % save_name, end='')
        continue
    
    if is_calculate_PF:
        print('\rLast extrapolation time: %.1f [sec]' % (time_end-time_start))
    time_start = time.time()
    print('\renter the directory: [%s]' % file[:-8])
    PF_data = extrapolate_potential(b_cube[:,:,:,0], n3 = b_cube.shape[3])
    is_calculate_PF = True
    np.save(save_name, PF_data)
    print('\rPotential field data saved @ %s' % save_name)
    time_end = time.time()
    time.sleep(0.01)