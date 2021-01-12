import h5py


def convert(root_path, source, target, type="train"):

    f = h5py.File(f'{root_path}/cap_process_train.h5','r')


    for i in list(f[type]):
        case = f[f'{type}/{i}/data'][()]
        label = f[f'{type}/{i}/label'][()]
        study_id = i.split('_')[0]