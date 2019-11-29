import DAK_main as DAK
import numpy as np
import pandas as pd
from scipy import stats

# set the path of label data
label_path = './demo_data/pheno.txt'
# cov_path = '../application/LC_pathway/LC_pathway_cov.txt'

# set the aim paths of result and
result_path = './demo_data/p.txt'
pathway_npy_path = './demo_data/pathway_onehot'
batch_npy_path = './demo_data/batch'
batch_label_path = './demo_data/label'
# batch_cov_path = './demo_data/cov'


pathway_num = 10
indiv_num = 1000
batch_size = 50
max_path_len = 20000

# convert raw format SNP into one-hot coding
raw_path = '../application/LC_pathway'
for path_iter in range(pathway_num):
    geno = pd.read_csv('./demo_data/pathway_' + str(path_iter) + '.raw_geno.txt', sep='\t', header=None,
                       index_col=None)
    geno = geno.values
    gene_one_hot = DAK.one_hot_convert(geno)
    np.save(pathway_npy_path + '/pathway_' +
            str(path_iter) + '.npy', gene_one_hot)
    print('One hot conversion for pathway ' + str(path_iter))

# convert pathway to training batches
batch_index = range(0, indiv_num, batch_size)
label = pd.read_csv(label_path, sep='\t', header=0, index_col=None)
label = np.squeeze(label.values)

# cov = pd.read_csv(cov_path, sep='\t', header=0, index_col=None)
# cov = cov.values.astype(np.float)
# cov = stats.zscore(cov, axis=0)

# divided to batches
for i in range(len(batch_index) - 1):
    batch_seq = np.zeros(
        [pathway_num, batch_size, max_path_len, 3], dtype=np.int8)
    for path_iter in range(pathway_num):
        path_data_buf = np.load(
            pathway_npy_path + '/pathway_' + str(path_iter) + '.npy')
        # [N,len,3]
        path_data_buf_select = path_data_buf[batch_index[i]                                             :batch_index[i + 1], :, :]
        batch_seq[path_iter, :, :path_data_buf_select.shape[1],
                  :] = path_data_buf_select

    batch_seq = batch_seq.astype(np.int8)
    np.save(batch_npy_path + '/batch_' + str(i) + '.npy', batch_seq)

    batch_label = label[batch_index[i]:batch_index[i + 1]]
    np.save(batch_label_path + '/batch_' + str(i) + '.npy', batch_label)

    # batch_cov = cov[batch_index[i]:batch_index[i + 1], :]
    # np.save(batch_cov_path + '/batch_' + str(i) + '.npy', batch_cov)

    print('make batch %d' % i)

# training DAK and test pathway
DAK.train(batch_npy_path, batch_label_path, None, result_path,
          batch_num=len(batch_index) - 1, batch_size=batch_size, pathway_num=pathway_num, max_path_len=max_path_len)
