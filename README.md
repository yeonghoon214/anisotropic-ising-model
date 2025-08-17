# Monte Carlo Simulation of Anisotropic Ising Model-tutorial

이 튜토리얼은 기존의 Ising model에서 교환상수 J를 Jx, Jy로 나누어 Jy/Jx 비율에 따른 상전이 시점 변화를 몬테카를로 시뮬레이션을 통해 분석했다.  
본 Anisotropic Ising Model의 코드는 https://rajeshrinet.github.io/blog/2014/ising-model/ 의 Isiotropic Ising model을 기반으로 작성하였다.  



# Ising Model

Ising model은 물리학자 Ernst ising과 Wilhelm Lenz의 이름을 딴 것으로, 통계역학에서 강자성체를 설명하기 위한 수학적 모델이다.
이 모델의 원자 스핀은 두 가지 상태(+1 or –1)를 가질 수 있으며, 각 스핀은 이웃 스핀과 상호작용한다.
Ising model의 Hamiltonian은 다음과 같이 정의된다.

H=−J∑⟨ij⟩SiSj.

에너지를 최소화하기 위해 J에 따라 질서있게 정렬된다.   
J > 0 : Ferromagnetic(이웃 스핀이 같은 방향을 선호)  
J < 0 : Anti-Ferromagnetic(이웃 스핀이 반대 방향을 선호)  




# 몬테카를로 시뮬레이션

몬테카를로 시뮬레이션은 난수(랜덤 숫자)를 이용해 어떤 물리적, 공학적 문제를 확률적으로 묘사하는 방법이다.
복잡한 물리계(ex) spin들의 상호작용)는 정확한 해를 구하기 어렵다. 이러한 현상을 모델링하여 무작위로 반복해서 평균을 내면 실제 해와 점점 가까워진다.


몬테카를로 시뮬레이션을 하기 위해선 난수가 필요하며, python의 `numpy.random`를 이용해 난수를 생성할 것이다.
+1과 -1를 가진 난수 2500개를 생성하면 다음과 같다.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormapN = 50

state = 2*np.random.randint(2, size=(N,N))-1
cmap = ListedColormap(['RoyalBlue', 'orange'])
plt.imshow((state+1)//2, cmap=cmap)
plt.show()
```

<img width="628" height="592" alt="image" src="https://github.com/user-attachments/assets/680097a8-f5af-401c-9db2-55a14232da95" />




