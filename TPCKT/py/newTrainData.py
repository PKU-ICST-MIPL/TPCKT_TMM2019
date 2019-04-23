import lmdb

env = lmdb.open('../TrainData/XmediaData/xmedianet_train_pool5_2x/')

txn = env.begin()

print txn.stat()
'''
for key, value in txn.cursor():
	print(key, value)
'''
env.close()