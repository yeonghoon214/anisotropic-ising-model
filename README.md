# Monte Carlo Simulation of Anisotropic Ising Model-tutorial

이 튜토리얼은 기존의 Ising model에서 교환상수 J를 Jx, Jy로 나누어 Jy/Jx 비율에 따른 상전이 시점 변화를 몬테카를로 시뮬레이션을 통해 분석했다.
본 Anisotropic Ising Model의 코드는 https://rajeshrinet.github.io/blog/2014/ising-model/ 의 Isiotropic Ising model을 기반으로 작성하였다.  



# Ising Model

Ising model은 물리학자 Ernst ising과 Wilhelm Lenz의 이름을 딴 것으로, 통계역학에서 강자성체를 설명하기 위한 수학적 모델이다.
이 모델의 원자 스핀은 두 가지 상태(+1 or –1)를 가질 수 있으며, 각 스핀은 이웃 스핀과 상호작용한다.
Ising model의 Hamiltonian은 다음과 같이 정의된다.

H=−J∑⟨ij⟩SiSj.

에너지를 최소화하는 방향으로 J에 따라 질서있게 정렬된다.   
J > 0 : Ferromagnetic(이웃 스핀이 같은 방향을 선호)  
J < 0 : Anti-Ferromagnetic(이웃 스핀이 반대 방향을 선호)  




# Monte Carlo Simulation

몬테카를로 시뮬레이션은 난수(랜덤 숫자)를 이용해 어떤 물리적, 공학적 문제를 확률적으로 묘사하는 방법이다.
복잡한 물리계(ex) spin들의 상호작용)는 정확한 해를 구하기 어렵다. 이러한 현상을 모델링하여 무작위로 반복해서 평균을 내면 실제 해와 점점 가까워진다.


몬테카를로 시뮬레이션을 하기 위해선 난수가 필요하며, python의 `numpy.random`를 이용해 난수를 생성할 것이다.  
`state = np.random.randint(2, size=(N,N))` : 무작위로 [0, +1] 값을 갖는 N×N 스핀 격자를 만드는 코드  
0과 +1을 가진 난수 2500개를 생성하면 다음과 같다.

```python
import numpy as np
import matplotlib.pyplot as plt

N = 50
state = np.random.randint(2, size=(N,N))
plt.imshow(state, cmap="binary")
plt.show()
```

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/44891d9c-343a-4181-a51a-240c53109891" />




# 2-D Ising Model using Metropolis Monte Carlo

Metropolis 몬테카를로 방법을 이용해 2D Ising model를 구현하는 기본 아이디어는 다음과 같다.

- [+1, -1] 가진 spin을 무작위로 n × n 격자에 생성한다.  
  `state = 2*np.random.randint(2, size=(N,N))-1` 
- 무작위로 스핀 하나(Sab)를 선택한 후, 스위칭을 진행하고(ex: +1 -> -1) 인접한 4개의 스핀과 Δ E를 계산한다.
- Δ E > 0일 경우, 에너지가 줄어들었기에 스위칭 된 상태를 유지힌다.
- Δ E < 0일 경우, e^{-ΔE/k_BT} 볼츠만 분포에 따라 확률적으로 결정된다. 0과 1사이의 난수를 뽑고 e^{-ΔE/k_BT} 보다 작을 경우 스위칭 된 상태 유지, 클 경우 기존의 상태를 유지한다.
- e^{-ΔE/k_BT} 볼츠만 분포에 따라 결정되는 것을 Metropolis 알고리즘 방식이라고 한다. 이러한 규칙을 따르는 Metropolis 알고리즘은, 열적 요동(thermal fluctuation)을 나타내며 볼츠만 분포에 따른 올바른 평형 상태를 재현할 수 있도록 한다.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/be093bb3-16b0-40a8-a56d-b5d95b94ae85" />



```python
a = np.random.randint(0, N)
b = np.random.randint(0, N)
s =  config[a, b]
nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
cost = 2*s*nb
if cost < 0:
 s *= -1
elif rand() < np.exp(-cost*beta):
 s *= -1
config[a, b] = s
```

# Anisotropic Ising Model 

isotropic의 경우 방향과 상관없이 J가 같지만, Anisotropic의 경우 방향에 따라 J가 다르다. 즉, Jx ≠ Jy

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d60e6696-5de7-4207-9572-810e9b325cf5" />

