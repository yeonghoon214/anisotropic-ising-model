# Monte Carlo Simulation of Anisotropic Ising Model-tutorial

이 튜토리얼은 기존의 Ising model에서 교환상수 J를 Jx, Jy로 나누어 Jy/Jx 비율에 따른 상전이 시점 변화를 몬테카를로 시뮬레이션을 통해 분석하였다.
본 Anisotropic Ising Model의 코드는 https://rajeshrinet.github.io/blog/2014/ising-model/ 의 Isiotropic Ising model을 기반으로 작성하였다.  



# Ising Model

Ising model은 물리학자 Ernst ising과 Wilhelm Lenz의 이름을 딴 것으로, 통계역학에서 강자성체를 설명하기 위한 수학적 모델이다.
이 모델의 원자 스핀은 두 가지 상태(+1 or –1)를 가질 수 있으며, 각 스핀은 이웃 스핀과 상호작용한다.
Ising model의 Hamiltonian은 다음과 같이 정의된다.

```math
\mathcal{H} = -J \sum_{\langle i, j \rangle} s_i s_j
```

에너지를 최소화하는 방향으로 J에 따라 질서있게 정렬된다.   
J > 0 : Ferromagnetic(이웃 스핀이 같은 방향을 선호)  
J < 0 : Anti-Ferromagnetic(이웃 스핀이 반대 방향을 선호)  




# Monte Carlo Simulation

몬테카를로 시뮬레이션은 난수(랜덤 숫자)를 활용하여 물리적·공학적 문제를 확률적으로 모델링하는 방법이다.
복잡한 물리계(ex: 스핀들의 상호작용)에서는 정확한 해를 직접 구하기 어렵지만, 무작위 시뮬레이션을 반복 수행하고 그 평균값을 취하면 실제 해에 점점 수렴하게 된다.

## 무작위 난수 생성

몬테카를로 시뮬레이션을 하기 위해선 난수가 필요하며, python의 `numpy.random`를 이용해 난수를 생성할 것이다.  
(`state = np.random.randint(2, size=(N,N))` : 무작위로 [0, 1] 값을 갖는 N×N 스핀 격자를 만드는 코드 )  
<br>
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




## 2-D Ising Model using Metropolis Monte Carlo

Metropolis 몬테카를로 방법을 이용해 2D Ising model를 구현하는 기본 아이디어는 다음과 같다.

- [+1, -1] 가진 spin을 무작위로 N × N 격자에 생성한다.  
  `state = 2*np.random.randint(2, size=(N,N))-1` 
- 무작위로 스핀 하나(Sab)를 선택한 후, 스위칭을 진행하고(ex: +1 -> -1) 인접한 4개의 스핀과 Δ E를 계산한다.
- Δ E > 0일 경우, 에너지가 줄어들었기에 스위칭 된 상태를 유지힌다.
- Δ E < 0일 경우, e^{-ΔE/k_BT} 볼츠만 분포에 따라 확률적으로 결정된다. 0과 1사이의 난수를 뽑고 e^{-ΔE/k_BT} 보다 작을 경우 스위칭 된 상태 유지, 클 경우 기존의 상태를 유지한다.
- e^{-ΔE/k_BT} 볼츠만 분포에 따라 결정되는 것을 Metropolis 알고리즘 방식이라고 한다. 이러한 규칙을 따르는 Metropolis 알고리즘은, 열적 요동(thermal fluctuation)을 나타내며 볼츠만 분포에 따른 올바른 평형 상태를 재현할 수 있도록 한다.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/be093bb3-16b0-40a8-a56d-b5d95b94ae85" /> <br>

<br>

```math

\mathcal{E} = −J⋅S_{ab}(S_{a−1,b}	+S_{a+1,b}	+S_{a,b+1} +S_{a,b−1}) 

```


  
```math
\mathcal{ΔE} = H_{new} - H_{old} = 2J⋅S_{ab}(S_{a−1,b}	+S_{a+1,b}	+S_{a,b+1} +S_{a,b−1})
```
<br>

Energy는 Ising model의 Hamiltonian에 따라 위와 같은 식으로 나타낼 수 있다.
위의 아이디어를 python으로 코드화한 것은 다음과 같다.  
( cost = ΔE , J = 1 , nb = 이웃한 4개의 spin)

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

## Anisotropic Ising Model 

isotropic의 경우 방향과 상관없이 J가 같지만, Anisotropic의 경우 방향에 따라 J가 다르다. 즉, Jx ≠ Jy

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d60e6696-5de7-4207-9572-810e9b325cf5" />

<br>

따라서, 우리는 Energy를 계산할 때, x방향과 y방향으로 나누어서 계산해야 한다.
<br>

```math

\mathcal{E_{anisotropic}} = −J_x⋅S_{ab}(S_{a−1,b}	+S_{a+1,b}) - J_y⋅S_{ab}(S_{a,b+1} +S_{a,b−1}) 

```
```math

\mathcal{ΔE_{anisotropic}} = 2⋅S_{ab}( J_x (S_{a−1,b}	+S_{a+1,b}) + J_y (S_{a,b+1} +S_{a,b−1})) 

```

<br>

Anisotropic model을 반영한 python 코드는 다음과 같다.

```python
a = np.random.randint(0, N)
b = np.random.randint(0, N)
s = config[a, b]
nb_x = config[(a+1)%N, b] + config[(a-1)%N, b]   
nb_y = config[a, (b+1)%N] + config[a, (b-1)%N]
cost = 2 * s * (Jx*nb_x + Jy*nb_y)
 if cost < 0:
   s *= -1
 elif rand() < np.exp(-cost * beta):
   s *= -1
 config[a, b] = s
```
<br>



<br>

## Monte Carlo Simulation Results

몬테카를로 시뮬레이션을 돌리기 위해서는 아래와 같은 파라미터들을 설정해야 한다.<br>

```python
nt      = 100          
N       = 16          
eqSteps = 1024        
mcSteps = 1024        
Jx, Jy  = -1.0, 0.5
```
- nt : 시뮬레이션에서 샘플링할 개수, x축(T)을 몇개의 점으로 나눌 것인지 결정
- N  : 격자 한변의 길이
- eqSteps : 평형으로 가기 위한 Monte Carlo steps (시스템을 초기 랜덤 상태에서 안정화시키기 위해)
- mcSteps : 평형 이후 실제 측정에 사용하는 Monte Carlo steps
- Jx, Jy  : 교환 상수 J를 변화시키며 상전이 시점을 분석


Monte Carlo step(MCStep) 한번은 N × N번 스핀을 무작위로 선택해 에너지 변화를 계산하고 평균을 내는 과정이다.<br>
관측값 하나를 얻기 위해서는 지정한 mcSteps 횟수(1024)만큼 이 과정을 반복해 평균을 낸다.
이렇게 얻은 값을 온도 구간을 나눈 개수(nt)만큼 반복하면 시뮬레이션이 완료된다.  

nt, N, mcSteps 값을 늘릴수록 결과가 실제 해(정확한 값)에 가까워지지만, 그만큼 시뮬레이션 속도는 느려진다는 단점이 있다.
추가적으로, Jx < 0 , Jy > 0 로 설정하여 x축은 Anti-Ferromagnetic을 선호하고 y축은 Ferromagnetic을 선호하게 모델링하였다.  


여기서 분석하는 물리량은 에너지(E), 비열(C), 자화(M), 자기 감수율(X)이며 공식은 다음과 같다.<br>  

```math
\mathcal{\langle  E \rangle} = \frac{1}{N} \sum_{\langle i \rangle N} H_i
```
```math
\mathcal{\langle  M \rangle} = \frac{1}{N} \sum_{\langle i \rangle N} S_i
```
```math
\mathcal{C} = \frac{B}{T} ( \langle  E^2 \rangle - \langle  E \rangle^2)
```
```math
\mathcal{X} = \frac{B}{T} ( \langle  M^2 \rangle - \langle  M \rangle^2)
```
<br> 

비열 C의 공식에서 $( \langle E^2 \rangle - \langle E \rangle^2 )$는 에너지의 분산을 나타낸다. 따라서 C는 T에 따른 에너지의 분산임을 알 수 있으며 자기감수율 X도 이와 같이 T에 따른 자화의 분산이다.
<br>

이제 $J_x = -1 , J_y = 0.75$로 교환 상수를 설정하고 몬테카를로 시뮬레이션을 진행하면 결과는 다음과 같다.  

<img width="400" height="720" alt="image" src="https://github.com/user-attachments/assets/63c50447-8e9d-4287-a36a-1be5365c7ec3" />  
<br>

- 에너지 그래프는 온도가 증가할수록 에너지가 점차 증가하여 시스템이 점점 불안정해짐을 보여준다.

- 비열 그래프는 온도 증가에 따라 상승하다가 특정 온도에서 최대값(peak)을 찍은 뒤 다시 감소하는데, 이는 해당 온도에서 에너지의 요동(분산)이 가장 크다는 것을 의미한다.
따라서 이 지점이 바로 상전이가 일어나는 임계 온도임을 확인할 수 있다.

다음으로 Jy / |Jx| 의 비율에 따른 상전이 시점 변화를 분석해보겠다.  
<br>
<img width="2074" height="902" alt="image" src="https://github.com/user-attachments/assets/1110ed5b-a33b-4b5d-8af1-3718093b5fae" />
 


- Jy / |Jx| 의 비율이 증가할수록 비열 그래프의 peak가 오른쪽으로 shift하는 것을 확인할 수 있다.
- Jy / |Jx| 의 비율이 증가할수록 상전이가 일어나는 Transition Temperature가 증가한다는 것을 알 수 있다.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/943b00c2-c947-4912-8fb7-b2224709d84f" />

- 위의 그래프를 통해 더 직관적으로 이해할 수 있다.
- Jy / |Jx| 의 비율이 증가할수록 Transition Temperature가 선형적으로 증가하는것을 보여준다.

- Jy / Jx = -0.75
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/91da076a-b343-44af-800b-3d25bcc37663" />

<br>

- Jy / Jx = 0.75
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/4b139704-f275-4164-b2d1-671cfbaf63e8" />

<br>

- T < Transition Temperature
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/814a9e04-7926-4e0d-bcc2-49bc25818aed" />

<br>

- T > Transition Temperature
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/0d4c353b-4054-479b-97b9-6840a96ecb21" />

  
![ising (2)](https://github.com/user-attachments/assets/90f22188-d600-4b87-b536-6bb5e12a2a14)

