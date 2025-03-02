# Our Federated Learning (FedHD) 

### Parameter List

**Datasets**: MNIST, Cifar-10, FEMNIST, Fashion-MNIST, Shakespeare.

**Model**: CNN, MLP, LSTM for Shakespeare

You can run like this:

python main.py --gpu 0 --dataset cifar --iid --epochs 200
python main.py --gpu 0 --dataset cifar --epochs 200
action='store_true'是用于指定参数的解析动作,当脚本中添加该参数时,它的值为True,否则为False。
