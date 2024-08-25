import numpy as np
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
from Model import PretrainModel, FewshotModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from opts import parse_args
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

args = parse_args()
labels = pickle.load(open('D:\\code\\Ours\\few-shot\\pretrain\\bypass-肺炎.pkl', "rb"))
labels = torch.from_numpy(labels).float()  # 或者使用 .long() 如果你的标签是整数类型


with open('D:\\code\\Ours\\few-shot\\pretrain\\connections.txt', 'rb') as f:
    adj_lists = eval(f.read())


# 读取患者特征
patient_features = pd.read_excel('D:\\code\\Ours\\few-shot\\pretrain\\患者自身特征.xlsx').values

print("预训练患者自身特征大小", patient_features.shape)

# 加载不同阶段的数据
preop_data_df = pd.read_excel('D:\\code\\Ours\\few-shot\\pretrain\\手术前信息.xlsx')
intraop_data_df = pd.read_excel('D:\\code\\Ours\\few-shot\\pretrain\\手术中信息.xlsx')
postop_data_df = pd.read_excel('D:\\code\\Ours\\few-shot\\pretrain\\手术后信息.xlsx')

# 确定疾病节点的数量
all_disease_nodes = set()
for patient_neighbors in adj_lists.values():
    all_disease_nodes.update(patient_neighbors)


# 创建疾病节点的One-Hot编码特征
disease_features = np.eye(len(all_disease_nodes))

print("预训练疾病节点特征大小", disease_features.shape)

# 如果患者特征的维度大于疾病节点的数量，添加零填充
if patient_features.shape[1] > disease_features.shape[1]:
    padding = np.zeros((disease_features.shape[0], patient_features.shape[1] - disease_features.shape[1]))
    disease_features = np.concatenate((disease_features, padding), axis=1)

# 如果疾病节点的数量大于患者特征的维度，添加零填充
if disease_features.shape[1] > patient_features.shape[1]:
    padding = np.zeros((patient_features.shape[0], disease_features.shape[1] - patient_features.shape[1]))
    patient_features = np.concatenate((patient_features, padding), axis=1)

num_patients = patient_features.shape[0]

print("num_patients:", patient_features.shape)

indices = torch.randperm(num_patients).tolist()
train_size = int(0.8 * num_patients)
train_patient_indices = set(indices[:train_size])
test_patient_indices = set(indices[train_size:])


# 合并患者特征和疾病特征
features = np.concatenate((patient_features, disease_features), axis=0)
feature_dim = features.shape[1]


all_features_set = set(preop_data_df.columns) | set(intraop_data_df.columns) | set(postop_data_df.columns)

all_features_ordered = OrderedDict.fromkeys(preop_data_df.columns.tolist() + intraop_data_df.columns.tolist() + postop_data_df.columns.tolist())

feature_to_index = {feature: idx for idx, feature in enumerate(all_features_ordered)}

time_steps = 3  # 时间步骤数量为3（术前、术中、术后）
# num_patients = preop_data_df.shape[0]
num_features = len(all_features_set)
patients_temporal_features_tensor = torch.zeros(num_patients, time_steps, num_features)

for time_step, data_df in enumerate([preop_data_df, intraop_data_df, postop_data_df]):
    for feature in data_df.columns:
        index = feature_to_index[feature]
        patients_temporal_features_tensor[:, time_step, index] = torch.tensor(data_df[feature].values)


train_lstm = patients_temporal_features_tensor[list(train_patient_indices)]
test_lstm = patients_temporal_features_tensor[list(test_patient_indices)]


patient_features_fsl_query = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\患者自身特征-query.xlsx').values
patient_features_fsl_support1 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\患者自身特征-support1.xlsx').values
patient_features_fsl_support2 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\患者自身特征-support2.xlsx').values

print("patient_features_fsl_query大小", patient_features_fsl_query.shape)

with open('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\邻接矩阵query.txt', 'rb') as f:
    adj_lists_fsl_query = eval(f.read())

with open('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\邻接矩阵support1.txt', 'rb') as f:
    adj_lists_fsl_support1 = eval(f.read())

with open('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\邻接矩阵support2.txt', 'rb') as f:
    adj_lists_fsl_support2 = eval(f.read())

preop_data_df_fsl_support1 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\心跳骤停或室颤-手术前信息-support1.xlsx')
intraop_data_df_fsl_support1 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\心跳骤停或室颤-手术中信息-support1.xlsx')
postop_data_df_fsl_support1 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\心跳骤停或室颤-手术后信息-support1.xlsx')


preop_data_df_fsl_support1_healthy = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\健康-手术前信息-support1.xlsx')
intraop_data_df_fsl_support1_healthy = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\健康-手术后信息-support1.xlsx')
postop_data_df_fsl_support1_healthy = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\健康-手术后信息-support1.xlsx')

preop_data_df_fsl_support2 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\心跳骤停或室颤-手术前信息-support2.xlsx')
intraop_data_df_fsl_support2 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\心跳骤停或室颤-手术中信息-support2.xlsx')
postop_data_df_fsl_support2 = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\心跳骤停或室颤-手术后信息-support2.xlsx')

preop_data_df_fsl_support2_healthy = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\健康-手术前信息-support2.xlsx')
intraop_data_df_fsl_support2_healthy = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\健康-手术后信息-support2.xlsx')
postop_data_df_fsl_support2_healthy = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\健康-手术后信息-support2.xlsx')


preop_data_df_fsl_query = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\手术前信息-query.xlsx')
intraop_data_df_fsl_query = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\手术中信息-query.xlsx')
postop_data_df_fsl_query = pd.read_excel('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\手术后信息-query.xlsx')

num_sick_support1 = preop_data_df_fsl_support1_healthy.shape[0]

num_sick_support2 = preop_data_df_fsl_support2_healthy.shape[0]


all_disease_nodes_fsl_query = set()
for patient_neighbors in adj_lists_fsl_query.values():
    all_disease_nodes_fsl_query.update(patient_neighbors)


all_disease_nodes_fsl_support1 = set()
for patient_neighbors in adj_lists_fsl_support1.values():
    all_disease_nodes_fsl_support1.update(patient_neighbors)

all_disease_nodes_fsl_support2 = set()
for patient_neighbors in adj_lists_fsl_support2.values():
    all_disease_nodes_fsl_support2.update(patient_neighbors)


disease_features_fsl_query = np.eye(len(all_disease_nodes_fsl_query))
disease_features_fsl_support1 = np.eye(len(all_disease_nodes_fsl_support1))
disease_features_fsl_support2 = np.eye(len(all_disease_nodes_fsl_support2))


if patient_features_fsl_query.shape[1] > disease_features_fsl_query.shape[1]:
    padding = np.zeros((disease_features_fsl_query.shape[0], patient_features_fsl_query.shape[1] - disease_features_fsl_query.shape[1]))
    disease_features_fsl_query = np.concatenate((disease_features_fsl_query, padding), axis=1)

if disease_features_fsl_query.shape[1] > patient_features_fsl_query.shape[1]:
    padding = np.zeros((patient_features_fsl_query.shape[0], disease_features_fsl_query.shape[1] - patient_features_fsl_query.shape[1]))
    patient_features_fsl_query = np.concatenate((patient_features_fsl_query, padding), axis=1)

if patient_features_fsl_support1.shape[1] > disease_features_fsl_support1.shape[1]:
    padding = np.zeros((disease_features_fsl_support1.shape[0], patient_features_fsl_support1.shape[1] - disease_features_fsl_support1.shape[1]))
    disease_features_fsl_support1 = np.concatenate((disease_features_fsl_support1, padding), axis=1)

if disease_features_fsl_support1.shape[1] > patient_features_fsl_support1.shape[1]:
    padding = np.zeros((patient_features_fsl_support1.shape[0], disease_features_fsl_support1.shape[1] - patient_features_fsl_support1.shape[1]))
    patient_features_fsl_support1 = np.concatenate((patient_features_fsl_support1, padding), axis=1)

if patient_features_fsl_support2.shape[1] > disease_features_fsl_support2.shape[1]:
    padding = np.zeros((disease_features_fsl_support2.shape[0], patient_features_fsl_support2.shape[1] - disease_features_fsl_support2.shape[1]))
    disease_features_fsl_support2 = np.concatenate((disease_features_fsl_support2, padding), axis=1)

if disease_features_fsl_support2.shape[1] > patient_features_fsl_support2.shape[1]:
    padding = np.zeros((patient_features_fsl_support2.shape[0], disease_features_fsl_support2.shape[1] - patient_features_fsl_support2.shape[1]))
    patient_features_fsl_support2 = np.concatenate((patient_features_fsl_support2, padding), axis=1)


num_patients_fsl_query = patient_features_fsl_query.shape[0]
num_patients_fsl_support1 = patient_features_fsl_support1.shape[0]
num_patients_fsl_support2 = patient_features_fsl_support2.shape[0]


indices_fsl_query = torch.randperm(num_patients_fsl_query).tolist()
train_size_fsl_query = int(0.6 * num_patients_fsl_query)
train_patient_indices_fsl = set(indices_fsl_query[:train_size_fsl_query])
test_patient_indices_fsl = set(indices_fsl_query[train_size_fsl_query:])


patient_indices_fsl_support1 = list(range(num_patients_fsl_support1))
patient_indices_fsl_support2 = list(range(num_patients_fsl_support2))


features_fsl_query = np.concatenate((patient_features_fsl_query, disease_features_fsl_query), axis=0)
feature_dim_fsl_query = features_fsl_query.shape[1]

features_fsl_support1 = np.concatenate((patient_features_fsl_support1, disease_features_fsl_support1), axis=0)
feature_dim_fsl_support1 = features_fsl_support1.shape[1]

features_fsl_support2 = np.concatenate((patient_features_fsl_support2, disease_features_fsl_support2), axis=0)
feature_dim_fsl_support2 = features_fsl_support2.shape[1]

all_features_set_fsl = set(preop_data_df_fsl_support1.columns) | set(intraop_data_df_fsl_support1.columns) | set(postop_data_df_fsl_support1.columns)


all_features_ordered_fsl = OrderedDict.fromkeys(all_features_set_fsl)

feature_to_index_fsl = {feature: idx for idx, feature in enumerate(all_features_ordered_fsl)}

num_features_fsl = len(all_features_set_fsl)

patients_temporal_features_tensor_fsl = torch.zeros(num_patients_fsl_query, time_steps, num_features_fsl)


def merge_patient_features(preop_df, intraop_df, postop_df, feature_to_index, num_features, time_steps):
    num_patients = preop_df.shape[0]
    patient_features = torch.zeros(num_patients, time_steps, num_features)

    for time_step, data_df in enumerate([preop_df, intraop_df, postop_df]):
        for feature in data_df.columns:
            if feature in feature_to_index:
                index = feature_to_index[feature]
                patient_features[:, time_step, index] = torch.tensor(data_df[feature].values)

    return patient_features


def build_support_set(sick_preop, sick_intraop, sick_postop, healthy_preop, healthy_intraop, healthy_postop,
                      num_samples=5, num_healthy=336):
    sick_features = merge_patient_features(sick_preop, sick_intraop, sick_postop, feature_to_index_fsl, num_features_fsl, time_steps)
    healthy_features = merge_patient_features(healthy_preop, healthy_intraop, healthy_postop, feature_to_index_fsl, num_features_fsl, time_steps)

    sick_indices = random.sample(range(sick_features.shape[0]), num_samples)
    healthy_indices = random.sample(range(healthy_features.shape[0]), num_samples)

    support_set = torch.zeros(num_samples, 2, time_steps + 1, num_features_fsl)

    for i in range(num_samples):
        support_set[i, 0, :-1, :] = sick_features[sick_indices[i]]
        support_set[i, 0, -1, 0] = sick_indices[i] + num_healthy  # 患病样本编号

        support_set[i, 1, :-1, :] = healthy_features[healthy_indices[i]]
        support_set[i, 1, -1, 0] = healthy_indices[i]  # 健康样本编号（加上患病样本的数量）

    return support_set


def build_support2(sick_preop, sick_intraop, sick_postop, healthy_preop, healthy_intraop, healthy_postop, num_samples=100, num_healthy=224):
    sick_features = merge_patient_features(sick_preop, sick_intraop, sick_postop, feature_to_index_fsl, num_features_fsl, time_steps)
    healthy_features = merge_patient_features(healthy_preop, healthy_intraop, healthy_postop, feature_to_index_fsl, num_features_fsl, time_steps)

    repeated_sick_indices = (list(range(sick_features.shape[0])) * (num_samples // sick_features.shape[0]))[:num_samples]
    healthy_indices = random.sample(range(healthy_features.shape[0]), num_samples)

    support_set = torch.zeros(num_samples, 2, time_steps + 1, num_features_fsl)

    for i in range(num_samples):
        support_set[i, 0, :-1, :] = sick_features[repeated_sick_indices[i]]
        support_set[i, 0, -1, 0] = repeated_sick_indices[i] + num_healthy  # 患病样本编号（加上健康样本的数量）

        support_set[i, 1, :-1, :] = healthy_features[healthy_indices[i]]
        support_set[i, 1, -1, 0] = healthy_indices[i]  # 健康样本编号

    return support_set

supports = [build_support_set(preop_data_df_fsl_support1, intraop_data_df_fsl_support1, postop_data_df_fsl_support1,
                                  preop_data_df_fsl_support1_healthy, intraop_data_df_fsl_support1_healthy,
                                  postop_data_df_fsl_support1_healthy)
                for _ in range(200)]

support2 = build_support2(preop_data_df_fsl_support2, intraop_data_df_fsl_support2, postop_data_df_fsl_support2,
                                  preop_data_df_fsl_support2_healthy, intraop_data_df_fsl_support2_healthy,
                                  postop_data_df_fsl_support2_healthy, num_samples=210)

for time_step, data_df in enumerate([preop_data_df_fsl_query, intraop_data_df_fsl_query, postop_data_df_fsl_query]):
    for feature in data_df.columns:
        index = feature_to_index_fsl[feature]
        patients_temporal_features_tensor_fsl[:, time_step, index] = torch.tensor(data_df[feature].values)


train_lstm_fsl = patients_temporal_features_tensor_fsl[list(train_patient_indices_fsl)]
test_lstm_fsl = patients_temporal_features_tensor_fsl[list(test_patient_indices_fsl)]


def total_pretrain_loss(zx, labels, zxr, zxc, zc, args):
    supervised_criterion = nn.BCEWithLogitsLoss()
    Lsup = supervised_criterion(zx, labels.float())

    zcaus = zxr + zxc

    causal_criterion = nn.BCEWithLogitsLoss()
    Lcaus = causal_criterion(zcaus, labels.float())

    uniform_criterion = nn.MSELoss()
    yuniform = torch.full((zx.size(0), 1), 0.5, device=zx.device)
    Luni = uniform_criterion(zc, yuniform)

    total_loss = Lsup + args.lambda1 * Lcaus + args.lambda3 * Luni
    return total_loss


model = PretrainModel(gcn_data=features, gcn_dim=feature_dim, gcn_out_dim=args.gcn_out_dim, adj_lists=adj_lists, lstm_input_dim=num_features,
                      hidden_dim=args.train_enc_dim, num_heads=args.num_heads, lstm_layers=args.lstm_layers, trans_layers=args.trans_layers, embed_dim=args.embed_dim, causal_dim=args.causal_dim, combine_method_xc='add')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_pre)

scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

def train(model, optimizer, train_data_gcn, train_data_lstm, labels, args, epochs=args.epochs):
    model.train()

    for epoch in range(epochs):
        permutation = torch.randperm(train_data_gcn.size(0))
        for i in range(0, train_data_gcn.size(0), args.batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + args.batch_size]
            batch_gcn = train_data_gcn[indices]
            batch_lstm = train_data_lstm[indices]
            batch_labels = labels[indices]

            zx, zc, zxr, zxc = model(batch_gcn, batch_lstm)

            loss = total_pretrain_loss(zx, batch_labels, zxr, zxc, zc, args)
            loss.backward()
            optimizer.step()

            scheduler.step()

            predictions = torch.sigmoid(zx).round()
            acc = accuracy_score(batch_labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())

            print(f'Epoch {epoch + 1}/{epochs}, Batch {i // args.batch_size + 1} - Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

def test(model, test_data_gcn, test_data_lstm, labels):
    model.eval()
    with torch.no_grad():
        _, _, zxr, zxc = model(test_data_gcn, test_data_lstm)
        probabilities = torch.sigmoid(zxr+zxc).cpu().numpy()
        predictions = probabilities.round()

        acc = accuracy_score(labels.cpu(), predictions)  # 移除predictions的.cpu()调用
        precision = precision_score(labels.cpu(), predictions, zero_division=0)
        recall = recall_score(labels.cpu(), predictions, zero_division=0)
        f1 = f1_score(labels.cpu(), predictions, zero_division=0)

        print(f'Test - Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}')


features_tensor = torch.FloatTensor(features)


train_patient_indices_list = list(train_patient_indices)

train_labels = labels[train_patient_indices_list]

test_patient_indices_list = list(test_patient_indices)

test_labels = labels[test_patient_indices_list]


train_patient_indices_tensor = torch.tensor(list(train_patient_indices))
test_patient_indices_tensor = torch.tensor(list(test_patient_indices))

save_path = 'D:\\code\\Ours\\parameters\\new_parameters.pth'
train(model, optimizer, train_patient_indices_tensor, train_lstm, train_labels, args)
torch.save(model.state_dict(), save_path)

model.load_state_dict(torch.load(save_path))
test(model, test_patient_indices_tensor, test_lstm, test_labels)

model_fsl = FewshotModel(gcn_data=features_fsl_query, gcn_dim=feature_dim_fsl_query, gcn_out_dim=args.gcn_out_dim, adj_lists=adj_lists_fsl_query, lstm_input_dim=num_features_fsl,
                         hidden_dim=args.train_enc_dim, num_heads=args.num_heads, lstm_layers=args.lstm_layers, trans_layers=args.trans_layers, embed_dim=args.embed_dim, causal_dim=args.causal_dim, combine_method_xc='add')

model_fsl.load_state_dict(torch.load(save_path), strict=False)
optimizer_fsl = torch.optim.Adam(model_fsl.parameters(), lr=args.lr_ft)


def finetune_loss(zxr, zxc, labels, args):
    labels_index = torch.argmax(labels, dim=1)

    supervised_criterion = nn.CrossEntropyLoss()

    Lsup_zxr = supervised_criterion(zxr, labels_index)
    Lsup_zxc = supervised_criterion(zxc, labels_index)

    total_loss = Lsup_zxr + Lsup_zxc

    return total_loss


def few_shot_train(model, optimizer, supports, train_data_gcn, train_data_lstm, labels, args):
    model.train()

    for epoch in range(args.epochs_ft):
        support_set = supports[epoch % len(supports)]
        last_time_step_info = support_set[:, :, -1, :]

        permutation = torch.randperm(train_data_gcn.size(0))
        total_loss = 0
        num_batches = 0

        for i in range(0, train_data_gcn.size(0), args.batch_size):
            indices = permutation[i:i + args.batch_size]
            batch_gcn = train_data_gcn[indices]
            batch_lstm = train_data_lstm[indices]
            batch_labels = labels[indices]

            optimizer.zero_grad()

            h_xr_batch, h_xc_batch = model.causal_model(batch_gcn, batch_lstm)

            class_means_xr, class_means_xc = model.process_support_set(support_set)

            zxr_batch, zxc_batch = model.compute_similarity(h_xr_batch, h_xc_batch, class_means_xr, class_means_xc)

            loss = finetune_loss(zxr_batch, zxc_batch, batch_labels, args)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_loss}')


def few_shot_test(model, support_set, test_data_gcn, test_queries, labels):
    model.eval()
    class_means_xr, class_means_xc = model.process_support_set(support_set)
    last_time_step_info = support_set[:, :, -1, :]
    with torch.no_grad():
        gcn_data = test_data_gcn
        lstm_data = test_queries
        h_xr_batch, h_xc_batch = model.causal_model(gcn_data, lstm_data)
        zxr_batch, zxc_batch = model.compute_similarity(h_xr_batch, h_xc_batch, class_means_xr, class_means_xc)
        zxr_batch_norm = torch.softmax(zxr_batch, dim=1)
        zxc_batch_norm = torch.softmax(zxc_batch, dim=1)
        probabilities = (zxr_batch_norm[:, 0] + zxc_batch_norm[:, 0]) / 2

        predictions = (probabilities < 0.5).int()


    labels_indices = torch.argmax(labels, dim=1)


    acc = accuracy_score(labels_indices.cpu(), predictions.cpu())
    precision = precision_score(labels_indices.cpu(), predictions.cpu(), pos_label=0)
    recall = recall_score(labels_indices.cpu(), predictions.cpu(), pos_label=0)
    f1 = f1_score(labels_indices.cpu(), predictions.cpu(), pos_label=0)

    cm = confusion_matrix(labels_indices.cpu(), predictions.cpu())
    tn, fp, fn, tp = cm[1][1], cm[1][0], cm[0][1], cm[0][0]

    fpr_value = fp / (fp + tn)

    fpr, tpr, thresholds = roc_curve(labels_indices.cpu(), probabilities, pos_label=0)  # 假设类别1是我们关注的类别
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Low Cardiac Output ')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_心跳骤停或室颤.png')
    plt.show()

    np.savetxt('roc_data_心跳骤停或室颤.csv', np.column_stack((fpr, tpr, thresholds)), delimiter=',', header='FPR,TPR,Thresholds')

    print(f'Test - Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, AUC: {roc_auc}')
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, FPR: {fpr_value:.4f}')



def load_labels(file_path):
    # 假设每行代表一个样本的标签，标签使用两列的One-Hot编码
    labels_df = pd.read_excel(file_path)
    labels = labels_df.iloc[:, :2]  # 假设前两列是One-Hot编码的标签
    return torch.from_numpy(labels.values).float()  # 转换为torch tensor，使用float类型

few_shot_labels = load_labels('D:\\code\\Ours\\few-shot\\fewshot\\心跳骤停或室颤\\query标签.xlsx')

train_patient_indices_list_fsl = list(train_patient_indices_fsl)

train_labels_fsl = few_shot_labels[train_patient_indices_list_fsl]

test_patient_indices_list_fsl = list(test_patient_indices_fsl)

test_labels_fsl = few_shot_labels[test_patient_indices_list_fsl]

train_patient_indices_tensor_fsl = torch.tensor(list(train_patient_indices_fsl))
test_patient_indices_tensor_fsl = torch.tensor(list(test_patient_indices_fsl))

few_shot_train(model_fsl, optimizer_fsl, supports, train_patient_indices_tensor_fsl, train_lstm_fsl, train_labels_fsl, args)

few_shot_test(model_fsl, support2, test_patient_indices_tensor_fsl, test_lstm_fsl, test_labels_fsl)

