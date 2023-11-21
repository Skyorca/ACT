#### 工作流

main入口： /exp/pipelines/run_framework.py
phase 1. 在源域上训练模型
phase 2. 训练adaptation模型
phase 3. 用isolation forest给目标域打伪标签

#### 数据格式

mat格式：
utils/data.py两个函数读取

def load_mat(file_dir, fn):
    fp = join(file_dir, fn)
    data = sio.loadmat(fp)
    return {
        "features": sp.lil_matrix(data['Attributes']),
        "adj": sp.csr_matrix(data['Network']),
        "ad_labels": np.squeeze(np.array(data['Label']))
    }



def mat_to_pyg_data(data, undirected=True):
    features = torch.from_numpy(data["features"].todense()).float()
    adj = data["adj"]
    edge_index, _ = from_scipy_sparse_matrix(adj)
    ad_labels = data['ad_labels']
    if undirected:
        print("Processing the graph as undirected...")
        if data.is_directed():
            edge_index = to_undirected(data.edge_index)
    data = Data(x=features, edge_index=edge_index)
    if undirected:
        assert data.is_undirected()
    return data, ad_labels



#### 包含的全部模型



#### 工作流文件




`phase 1` /exp/gdev_net_sup/gdev_net_sup.py
主要调用runners.g_dev_net_runner.GraphDevNetRunner加载数据和模型进行训练，模型：models.gnns.graph_dev_net.GraphDevNet

note1: 每次batch训练都从正负样本中分别均等采样然后计算loss
note2: 有梯度裁剪的操作
note3: 由于使用deviation loss，所以分类器输出一个未经激活的value, 分类层(n_dim,1) / 可换成BCElWithLogitsLoss

训练完之后source encoder存在了ckpt,模型的state_dict和优化器的state_dict 【ckpt/src_gdev/{dataset}_{time}/best.pt】

`phase 2` /exp/act/act.py
调用runners/act_runner.py
主要包含跨域负样本（正常节点）之间的sinkhorn loss和 目标域节点的对比学习loss
需要知道目标域节点的标签才行，这部分如何和伪标签算法联动？
note1: sinkhorn loss和对比loss分别优化，优化两次，而不是加在一起过后联合优化

note2: 这里直接用source model的分类层打分，而没有训练新的target打分层

加载phase1的source encoder，实例化新的target encoder

训练完之后存target encoder 【ckpt/act/{src}_to_{tgt}_{time}/unsup_act_{epoch}.pt】



Q: 是怎么做到target和source oneclass对齐的？

对齐的是随机的embedding而没有把类别信息加进去。



`phase 3` /exp/sl/sl.py

调用target encoder，利用IF打异常分数之后利用ranking找到可靠的正负样本伪标签集合，之后利用deviation loss优化target encoder，最后输出结果



三个阶段不是反复迭代的关系，是串行；phase3相当于一个引入外界的先验进行修正的过程



#### 改动

1. base runner去掉tensorboard
2. 
