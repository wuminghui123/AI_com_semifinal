# AI_com_semifinal
 1、运行环境 
	tensorflow2.1 2080ti显卡
	
 2、所需模型文件和其他一些文件位置
	链接：https://pan.baidu.com/s/1R09xEPV43F20rS67LkiKiA 
	提取码：1234 
	通过以上百度网盘下载我们的四个模型文件，将.model_32.best.h5和.model_8.best.h5文件拷贝至train_test文件夹中。
	将.model_fenlei_8.best.h5和.model_fenlei_32.best.h5文件拷贝至预分类文件夹。
	并将主办方自带的H.bin,H_val.bin,Pilot_16,Pilot_64,Y_1.csv,Y_2.csv分别拷贝至train_test和预分类这两个文件夹中。
	
 3、预分类
	运行预分类文件夹中get_XY.py文件生成分类样本
	运行train.py训练预分类网络，如果已经从百度网翻盘下载则不需要训练。
	运行test.py对测试集进行测试可以看到大部分mode都是0
	
 4、端到端解调
	打开train_test文件夹
	运行MyModel8.py和MyModel.py分别训练8导频和32导频的端到端解调网络，如果已经从百度网翻盘下载则不需要训练。
	运行Detect.py文件对所给测试集文件进行端到端解调Detect.py
	
 5、分集与合并
	多次修改model_define.py中网络定义的一些细节参数例如卷积核大小、通道数等参数，多次进行训练。得到的结果性能都差不多都在0.95上下并且都在我的提交记录中提交过。
	train_test和百度网盘中方给出的模型是其中性能最好的一个，运行train_test文件中的Detect.py文件得到的结果对应于 分集与最大似然合并文件夹中的压缩包中的X_pre_1_1.bin和X_pre_2_2.bin，是其中性能最好的
	这些独立的分集位于分集与最大似然合并问家中的压缩包，需要先解压缩
	将训练的多个网络端到端解调的信号看作是独立的分集信号，我们采用最大似然方式进行合并，运行分集与最大似然合并文件夹中的ML.py得到最终结果
	
 6、最终结果位置
	最终结果位于分集与最大似然合并文件夹中的X_pre_1.bin和X_pre_2.bin