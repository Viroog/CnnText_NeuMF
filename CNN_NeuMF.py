import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Embedding, Input, Multiply, Flatten, Concatenate, Lambda, Conv2D, MaxPool1D, \
    Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
from tensorflow.keras.metrics import RootMeanSquaredError
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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

    # CNN要用到的参数
    parser.add_argument('--conv_embedding_dim', type=int, default=128,
                        help='Embedding size of CnnText model')
    parser.add_argument('--reg_conv', type=int, default=0,
                        help='Regularization for CnnText Embedding')
    parser.add_argument('--filter_sizes', nargs='?', default='[3, 4, 5]',
                        help='The height size of three filters,')
    parser.add_argument('--filter_nums', type=int, default=100,
                        help='The nums of kernel')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='The rate of dropout')

    return parser.parse_args()


def get_CNN_NeuMf_data(num_words=5000):
    data = pd.read_csv('Data/游戏推荐数据/processed_data.csv', encoding='utf-8')

    # 去除空缺值
    data = data.dropna(axis=0, how='any')

    tokenizer = Tokenizer(num_words=5000)
    review = data['review'].tolist()
    tokenizer.fit_on_texts(review)

    num_vocabs = len(tokenizer.word_index) + 1

    # 由字符变成id
    review = tokenizer.texts_to_sequences(review)

    # 计算所有句子的平均长度
    total_len = sum(len(sub_review) for sub_review in review)
    avg_length = int(total_len / len(review))

    # 将所有句子的长度定为平均句子的长度，长的截取，短的补充
    review = pad_sequences(review, padding='post', maxlen=avg_length)
    # 替换原本数据的review
    data['review'] = review.tolist()

    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, 'item_id':'review'], data.loc[:, 'rating'],
                                                        test_size=0.2, shuffle=True, random_state=42)

    # 获取用户输入维度和物品输入维度
    num_users, num_items = data['user_id'].max(), data['item_id'].max()
    max_rating, min_rating = data['rating'].max(), data['rating'].min()

    # 转变成二维的
    X_train['review'] = X_train['review'].astype({'review': 'object'})
    arr_2d = X_train['review'].apply(lambda x: pd.Series(x)).to_numpy()

    # 训练集
    train_review_input, train_user_input, train_item_input, train_rating = \
        np.array(arr_2d), np.array(X_train['user_id']), \
            np.array(X_train['item_id']), np.array(y_train)

    # 转变成二维的
    X_test['review'] = X_test['review'].astype({'review': 'object'})
    arr_2d = X_test['review'].apply(lambda x: pd.Series(x)).to_numpy()
    # 测试集
    test_review_input, test_user_input, test_item_input, test_rating = \
        np.array(arr_2d), np.array(X_test['user_id']), \
            np.array(X_test['item_id']), np.array(y_test)

    return {
        'num_vocabs': num_vocabs,
        'num_users': num_users,
        'num_items': num_items,
        'review_length': avg_length,
        'train_review_input': train_review_input,
        'train_user_input': train_user_input,
        'train_item_input': train_item_input,
        'train_rating': train_rating,
        'test_review_input': test_review_input,
        'test_user_input': test_user_input,
        'test_item_input': test_item_input,
        'test_rating': test_rating,
        'max_rating': max_rating,
        'min_rating': min_rating
    }


def get_model(num_vocabs, num_users, num_items, mf_dim, layers, reg_layers, reg_mf, conv_embedding_dim, review_length,
              reg_conv, filter_sizes, filter_nums, dropout_rate, max_rating, min_rating):
    num_layer = len(layers)
    # 输入变量
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    review_input = Input(shape=(review_length,), name='review_input')

    # CnnText Embedding层
    Conv_Embedding = Embedding(input_dim=num_vocabs, output_dim=conv_embedding_dim, name='conv_embedding',
                               embeddings_regularizer=l2(reg_conv), input_length=review_length)

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

    # 根据单词从Embedding中抽取向量
    review_matrix = Conv_Embedding(review_input)
    # 在最后一维扩充一维，代表通道数
    # tensorflow中的格式，与pytorch不一样 pytorch在第一维，而tensorflow在最后一维
    review_matrix = tf.expand_dims(review_matrix, -1)

    pooled_outputs = []
    for (idx, filter_size) in enumerate(filter_sizes):
        conv = Conv2D(filters=filter_nums, kernel_size=[filter_size, conv_embedding_dim], activation='relu',
                      name=f'conv{idx}')
        # 卷积后的输出 shape:[None, review_length, 1, kernel_nums]
        conv_out = conv(review_matrix)
        # 一维池化，选出每一个kernel上的最大值
        # 压缩维度，把通道数这个维度，即维度2压缩掉
        conv_out = tf.squeeze(conv_out, 2)
        # 最大池化层
        pool = MaxPool1D(pool_size=review_length - filter_size + 1, name=f'max_pool{idx}')
        # 池化后的输出
        pooled_out = pool(conv_out)

        pooled_outputs.append(pooled_out)

    # 从第1维开始拼接起来，即最后形成 [None, 300]，第0维为batch
    review_vector = tf.concat(pooled_outputs, axis=2)
    # shape: [None, filter_nums * len(filter_sizes)]
    review_vector = tf.reshape(review_vector, shape=(-1, filter_nums * len(filter_sizes)))

    # 再经过一个dropout层
    dropout = Dropout(rate=dropout_rate)
    review_vector = dropout(review_vector)

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
    predict_vector = Concatenate(axis=-1)([review_vector, mf_vector, mlp_vector])

    # 输出层
    # 因为训练数据的评分已经被缩放至[0,1]，因此输出层的激活函数用sigmoid同样缩放到[0,1]，用softmax的效果有点差
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
    prediction = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(prediction)

    model = Model(inputs=[review_input, user_input, item_input],
                  outputs=prediction)

    return model


if __name__ == '__main__':
    args = parse_args()
    # 全局参数
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

    # CnnText参数
    conv_embedding_dim = args.conv_embedding_dim
    reg_conv = args.reg_conv
    filter_sizes = eval(args.filter_sizes)
    filter_nums = args.filter_nums
    dropout_rate = args.dropout_rate

    result = get_CNN_NeuMf_data()

    # 获取两个输入的维度
    num_vocabs, num_users, num_items = result['num_vocabs'], result['num_users'], result['num_items']
    max_rating, min_rating = result['max_rating'], result['min_rating']
    review_length = result['review_length']

    print(f'CNN_NeuMF arguments: {args}')
    model_out_file = f'model/CnnText_NeuMF_non_min_max.h5'

    # 获得CNN_NCF模型
    model = get_model(num_vocabs, num_users, num_items, mf_dim, layers, reg_layers, reg_mf, conv_embedding_dim,
                      review_length, reg_conv, filter_sizes, filter_nums, dropout_rate, max_rating, min_rating)
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
    train_review_input, train_user_input, train_item_input, train_rating = result['train_review_input'], result[
        'train_user_input'], result['train_item_input'], result['train_rating']

    # 测试集
    test_review_input, test_user_input, test_item_input, test_rating = result['test_review_input'], result[
        'test_user_input'], result['test_item_input'], result['test_rating']

    history = model.fit(x=[train_review_input, train_user_input, train_item_input],
                        y=train_rating,
                        batch_size=batch_size, epochs=num_epochs,
                        validation_split=0.2)

    # 保存模型参数
    model.save(model_out_file)

    # 利用模型进行预测
    pred_rating = model.predict(x=[test_review_input, test_user_input, test_item_input])

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
