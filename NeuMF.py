import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Embedding, Input, Multiply, Flatten, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras import backend as K
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

tf.compat.v1.disable_eager_execution()


def parse_args():
    parser = argparse.ArgumentParser(description='Run CNN_NeuMF.')

    # 全局参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='sgd',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')

    # NCF需要用的参数
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64, 32, 16, 8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. "
                             "So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0, 0, 0, 0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")

    return parser.parse_args()


# NeuMF模型需要的数据
def get_data():
    data = pd.read_csv('Data/游戏推荐数据/processed_data.csv', encoding='utf-8')

    data = data.dropna(axis=0, how='any')

    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, 'item_id':'user_id'], data.loc[:, 'rating'],
                                                        test_size=0.2, shuffle=True, random_state=42)

    # 获取用户输入维度和物品输入维度
    num_users, num_items = data['user_id'].max(), data['item_id'].max()
    max_rating, min_rating = data['rating'].max(), data['rating'].min()

    # 训练集
    train_user_input, train_item_input, train_rating = np.array(X_train['user_id']), np.array(
        X_train['item_id']), np.array(y_train)

    # 测试集
    test_user_input, test_item_input, test_rating = np.array(X_test['user_id']), np.array(
        X_test['item_id']), np.array(y_test)

    return {
        'num_users': num_users,
        'num_items': num_items,
        'train_user_input': train_user_input,
        'train_item_input': train_item_input,
        'train_rating': train_rating,
        'test_user_input': test_user_input,
        'test_item_input': test_item_input,
        'test_rating': test_rating,
        'max_rating': max_rating,
        'min_rating': min_rating
    }


def get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, max_rating, min_rating):
    num_layer = len(layers)
    # 输入变量
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # MF Embedding层
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_regularizer=l2(reg_mf), input_length=1)

    # MLP Embedding层
    # 输出是第0层的一半，用户电影各一半，拼接则为第一层的大小
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name='mlp_embedding_user',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    # MF
    # 扁平化
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    # 进行element-wise product，得到结果依然为向量 shape: [None, 8]
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # MLP
    # 扁平化
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    # 直接拼接，大小为第0层MLP的大小 shape: [None, 64]
    mlp_vector = Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])

    # MLP向前传播
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu', kernel_regularizer=l2(reg_layers[idx]), name=f'layer{idx}')
        mlp_vector = layer(mlp_vector)

    # 最终的vector，直接拼接mf_vector和mlp_vector shape: [None, 16]
    predict_vector = Concatenate(axis=-1)([mf_vector, mlp_vector])

    # 输出层
    # 因为训练数据的评分已经被缩放至[0,1]，因此输出层的激活函数用sigmoid同样缩放到[0,1]，用softmax的效果有点差
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
    prediction = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(prediction)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model


if __name__ == '__main__':
    args = parse_args()
    # 全局参数
    # train_path = args.train_path
    # test_path = args.test_path
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    optimizer = args.optimizer
    verbose = args.verbose

    # NCF参数
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)

    result = get_data()

    # 获取两个输入的维度
    num_users, num_items = result['num_users'], result['num_items']
    max_rating, min_rating = result['max_rating'], result['min_rating']

    print(f'CNN_NeuMF arguments: {args}')
    model_out_file = 'model/NeuMF_non_min_max.h5'

    # 获得普通的NCF模型
    model = get_model(num_users + 1, num_items + 1, mf_dim, layers, reg_layers, reg_mf, max_rating, min_rating)
    # 损失函数改为mse，因为mse能更快收敛
    if optimizer.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='mse',
                      metrics=[RootMeanSquaredError(name='rmse')])
    elif optimizer.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='mse',
                      metrics=[RootMeanSquaredError(name='rmse')])
    elif optimizer.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='mse',
                      metrics=[RootMeanSquaredError(name='rmse')])
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='mse',
                      metrics=[RootMeanSquaredError(name='rmse')])

    # 训练集
    train_user_input, train_item_input, train_rating = result['train_user_input'], result['train_item_input'], result[
        'train_rating']

    # 测试集
    test_user_input, test_item_input, test_rating = result['test_user_input'], result['test_item_input'], result[
        'test_rating']

    history = model.fit(x=[train_user_input, train_item_input],
                        y=train_rating,
                        batch_size=batch_size, epochs=num_epochs,
                        validation_split=0.2)

    # 保存模型参数
    model.save(model_out_file)

    # 利用模型进行预测
    pred_rating = model.predict(x=[test_user_input, test_item_input])

    df = pd.DataFrame({
        'pred': pred_rating.ravel(),
        'real': test_rating.ravel()
    })

    df.to_csv('预测结果.csv', encoding='utf-8', index=False)

    # 计算指标
    mse = mean_squared_error(test_rating, pred_rating)
    mae = mean_absolute_error(test_rating, pred_rating)
    r2 = r2_score(test_rating, pred_rating)

    print(f'test mse: {mse}, test mae: {mae}, test r2 score: {r2}')

    plt.plot(np.arange(1, num_epochs + 1), history.history['loss'], marker='o', label='loss')
    plt.plot(np.arange(1, num_epochs + 1), history.history['val_loss'], marker='o', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss pic')
    plt.legend(loc='best')
    plt.show()

    plt.plot(np.arange(1, num_epochs + 1), history.history['rmse'], marker='o', label='rmse')
    plt.plot(np.arange(1, num_epochs + 1), history.history['val_rmse'], marker='o', label='val rmse')
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.title('rmse pic')
    plt.legend(loc='best')
    plt.show()
