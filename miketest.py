from sklearn.neighbors import KNeighborsClassifier

def fun1(knnclass, **kw):
	print( knnclass(kw) )
	
fun1( KNeighborsClassifier , n_neighbors=2 )