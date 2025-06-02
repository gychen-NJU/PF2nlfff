from PF2nlfff.packages import *
from PF2nlfff.model_3D import *
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

min_dim = 10000
for ids in training_ds:
    test_dim = np.min(adc_shape(read_bcube(ids[0]), is_print_info=False).shape[1:])
    if min_dim>test_dim:
        min_dim = test_dim
print('minimum dimension: ', min_dim)

data_tr = []
pf_data = []
for ds in training_ds:
    for ids in ds:
        ipf   = np.load(ids[:-8]+'PF_data.npy')
        ibxyz = read_bcube(ids)
        shp   = adc_shape(ibxyz, is_print_info=False).shape[1:]
        i1,i2,i3 = (np.array(shp)//2-min_dim/2).astype(int)
        data_tr.append(ibxyz[:,i1:i1+80,i2:i2+80,0:80])
        pf_data.append(ipf[:,i1:i1+80,i2:i2+80,0:80])
data_tr = np.array(data_tr)
pf_data = np.array(pf_data)

gan_model = GAN_model(training_ds, 1,in_device='cuda:0',out_device='cuda:0',cut_size=min_dim)
LR = 1.e-6
Delta=200
gamma=0.998
gan_model.print_epoch=10
gan_model.save_epoch=1000
gan_model.is_cut = False
gan_model.keep_file = False
gan_model.save_models_dir = './GAN_models/'
gan_model.train(5001,lr=LR,Delta=Delta,gamma=gamma, batch_size=10, ipt=pf_data, opt=data_tr)