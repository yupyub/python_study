CIFAR10 을 이용한 학습 모델을 여러 종류 만들어 보기 위해
Training 환경과 Test 환경을 만들었다.
각각 CIFAR10_Train.py 과 CIFAR10_Test.py 이다.
각 프로그램을 실행하기 위해서는 NN_Model_1.py 와 같이 
Model을 만들어서 *.pt 파일로 저장해야 한다.
그리고 CIFAR10_Train.py를 실행하면, train된 모델이 
*_cp.tar 파일로 저장된다. 
성능 테스트를 원한다면 CIFAR10_Test.py 를 실행하면 된다.

(@@ 주의 @@)
각 파일의 실행위치에 대해 상대주소로 DATA 폴더가 지정된다
