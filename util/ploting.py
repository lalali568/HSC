import scienceplots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import statistics
import os, torch
import numpy as np
import datetime


def loss_recored(loss, model, dataset):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join('plot', 'loss_recored', model, dataset)
    os.makedirs(path, exist_ok=True)
    file_name = date_string + '.pdf'
    pdf = PdfPages(os.path.join(path, file_name))
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('loss')
    ax1.plot(range(len(loss)), loss)
    pdf.savefig(fig)
    plt.close()
    pdf.close()


plt.style.use(['science', 'ieee'])
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2


def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_out(y_true, y_pred, ascore,model,dataset,val=False,caption=None):
    #if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    if val:
        path = os.path.join('plot', 'out', model, dataset,"val")
    else:
        path = os.path.join('plot', 'out', model, dataset, "test")
    os.makedirs(path, exist_ok=True)
    file_name = date_string + '.pdf'
    pdf = PdfPages(os.path.join(path, file_name))

    for dim in range(y_true.shape[1]):
        y_t, y_p, a_s = y_true[:, dim], y_pred[:, dim], ascore[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 这个就是构造一个有两行一列的子图的图，同时两个图共享x轴
        ax1.set_ylabel('Value')
        if caption is not None:
            if dim ==0:
                ax1.set_title(f"{caption}  Dimension = {dim} ")
        else:
            ax1.set_title(f'Dimension = {dim} ')
        # if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
        ax1.plot(smooth(y_t), linewidth=0.4, label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
        if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()
    pdf.close()

def prediction_out(orig_pred,pred,gt,model,dataset):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join('plot', 'prediction_out', model, dataset)
    os.makedirs(path, exist_ok=True)
    file_name = date_string + '.pdf'
    pdf = PdfPages(os.path.join(path, file_name))
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
    ax1.plot(pred,linewidth=0.3,color='red',label='pred',linestyle='solid')
    ax1.plot(gt, linewidth=0.4, color='black', label='gt',linestyle='solid')
    ax1.plot(orig_pred, linewidth=0.4, color='blue', label='orig_pred',linestyle='solid')
    ax2.plot(pred,linewidth=0.3,color='red',label='pred')
    ax3.plot(gt, linewidth=0.4, color='black', label='gt',linestyle='solid')
    ax4.plot(orig_pred, linewidth=0.4, color='blue', label='orig_pred',linestyle='solid')
    ax1.legend()#把标签表示出来
    pdf.savefig(fig)
    plt.close()
    pdf.close()

#这个是看每个时间维度的误差和
def record_loss(l,loss,model,dataset,val=False):
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    if val:
        path = os.path.join('plot', 'loss_each_timestamp', model, dataset, "val")
    else:
        path = os.path.join('plot', 'loss_each_timestamp', model, dataset, "test")
    os.makedirs(path, exist_ok=True)
    #处理一下数据
    loss=np.sum(loss,axis=1)
    file_name = date_string + '.pdf'
    pdf = PdfPages(os.path.join(path,file_name))
    fig, ax1 = plt.subplots()
    ax1.plot(l,linewidth = 0.3,color='black',label='label for each timestamp')
    ax1.plot(loss,linewidth = 0.3,color='red',label='loss for each timestamp')
    ax1.legend()
    pdf.savefig(fig)
    plt.close()
    pdf.close()

def loss_eachtimestamp_prediction_out(ground,pred,loss,model,dataset,):
    ground= ground.squeeze()
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join('plot', 'loss_eachtimestamp_prediction_out', model, dataset)
    os.makedirs(path, exist_ok=True)
    #处理一下数据
    loss = loss.squeeze()
    if loss.ndim!=1:
        loss=np.sum(loss,axis=1)
        max_loss = np.max(loss)
        min_loss =np.min(loss)
        loss = (loss-min_loss)/(max_loss-min_loss)
    file_name = date_string + '.pdf'
    pdf = PdfPages(os.path.join(path,file_name))
    fig, ax1 = plt.subplots()
    ax1.plot(loss,linewidth = 0.3,color='red',label='loss for each timestamp')
    ax1.fill_between(range(len(loss)),0.5,0,where=(ground>0),label="ground_label",facecolor='red',alpha=0.3)
    ax1.fill_between(range(len(loss)),1,0.5,where=(pred>0),label="pred_label",facecolor='blue',alpha=0.3)
    ax1.legend()
    pdf.savefig(fig)
    plt.close()
    pdf.close()