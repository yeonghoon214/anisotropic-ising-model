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

예를 들어, Ising model을 모델링하면 다음과 같다.

- [+1, -1] 가진 spin을 무작위로 N × N 격자에 생성한다.  
- 무작위로 스핀 하나(Sab)를 선택한 후, 스위칭을 진행하고(ex: +1 -> -1) 인접한 4개의 스핀과 Δ E를 계산한다.(Hamiltonian식 이용)

```math

\mathcal{E} = −J⋅S_{ab}(S_{a−1,b}	+S_{a+1,b}	+S_{a,b+1} +S_{a,b−1}) 

```
```math
\mathcal{ΔE} = H_{new} - H_{old} = 2J⋅S_{ab}(S_{a−1,b}	+S_{a+1,b}	+S_{a,b+1} +S_{a,b−1})
```

- Δ E > 0일 경우, 에너지가 줄어들었기에 스위칭 된 상태를 유지힌다.
- Δ E < 0일 경우, exp{-ΔE/k_BT} 볼츠만 분포에 따라 확률적으로 결정된다. 0과 1사이의 난수를 뽑고 exp{-ΔE/k_BT} 보다 작을 경우 스위칭 된 상태 유지, 클 경우 기존의 상태를 유지한다.
  
  (exp{-ΔE/k_BT} 그래프를 통해 알 수 있듯이, T가 낮을 때는 스위칭 상태를 거절할 확률이 높으며, T가 증가할수록 스위칭 상태를 수용할 확률이 증가한다)
- e^{-ΔE/k_BT} 볼츠만 분포에 따라 결정되는 것을 Metropolis 알고리즘 방식이라고 한다. 이러한 규칙을 따르는 Metropolis 알고리즘은, 열적 요동(thermal fluctuation)을 나타내며 볼츠만 분포에 따른 올바른 평형 상태를 재현할 수 있도록 한다.

  <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/be093bb3-16b0-40a8-a56d-b5d95b94ae85" /> <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/9e0209d6-b827-4e8e-a208-924e5cf9588b" />


## 무작위 난수 생성

몬테카를로 시뮬레이션을 하기 위해선 난수가 필요하며, python의 `numpy.random`를 이용해 난수를 생성할 것이다.  
`state = np.random.randint(2, size=(N,N))` : 무작위로 0 이상 2미만, 즉 [0, 1] 값을 갖는 N×N 스핀 격자를 만드는 코드   
`state = 2*np.random.randint(2, size=(N,N))-1` : [0, 1]값에 2를 곱한 후 1을 뺴준다, 즉 [-1, +1] 값을 갖는 N×N 스핀 격자를 만드는 코드
`plt.imshow(state, cmap="binary")`: state를 이미지처럼 시각화해주는 함수, cmap="binary"은 흑백 컬러맵을 사용한다는 코드(흑: -1, 백: +1)  

<br>
-1과 +1을 가진 난수 2500개를 생성하면 다음과 같다.

```python
import numpy as np
import matplotlib.pyplot as plt

N = 50
state = 2*np.random.randint(2, size=(N,N))-1
plt.imshow(state, cmap="binary")
plt.show()
```

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/44891d9c-343a-4181-a51a-240c53109891" />




## 2-D Ising Model using Metropolis Monte Carlo code

위의 2-D Ising Model을 모델링한 python 코드는 다음과 같다.  
( cost = ΔE , J = 1 , nb = 이웃한 4개의 spin)

```python
def mcmove(config, beta, Jx, Jy):
    for i in range(N):
        for j in range(N):
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
    return config
```

`config`:  프로그램의 동작을 제어하는 변수 또는 값을 저장할 떄 사용
 - input[a, b] : 무작위로 각각 x,y축에서 0~N-1 값을 선택
 - output[s] : s에 무작위로 선택한 a,b 값을 저장
 - N×N 반복할 때마다 config는 새로운 값(위치)을 저장하여 에너지를 계산

   <br>
`rand() < np.exp(-cost*beta)` : Metropolis 알고리즘 구현 코드  
(`from numpy.random import rand`을 통해 rand()를 사용)  

   <br>
따라서, mcmove는 Metropolis 알고리즘을 이용하여 스핀들이 스위칭 될지 말지 결정하는 함수이다.




### boundary condition

  
  <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/721f7957-c936-499c-8948-5571ff330ae2" />

무작위로 선택한 스핀의 위치가 위의 사진처럼 경계에 있다면 인접하는 스핀의 개수는 4개가 아닌 3개가 된다.
이러한 문제를 해결하기 위해 다음과 같은 code를 사용했다. <br>

     
`nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]`:
 - x축 경계일 때(a=N-1, b=3) , config[((N-1)+1)%N,b] = config[0 ,b] 이므로 nb = config[0,3] + config[N-1,4] + config[(N-2),3] + config[N-1,2] 이다. 이것을 그림으로 나타내면 아래와 같다.
   
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/4566acf3-c56a-44e7-a820-c0c1fc9fedf6" />



   -> 오른쪽과 왼쪽, 위쪽과 아래쪽이 연결되어 있는 도넛과 같은 형태이다.
   
 - 경계가 아닐 경우, (a+1)&N = (a+1) 이므로 영향을 받지 않는다.
     
 



## Anisotropic 2D Ising Model 

Mahrous R. Ahmed 교수님은 xy 평면에서 Anti-ferromagnetic(J1) z축에서 Ferromagnetic(J2)의 상호작용을 갖는 Anisotropic 3D Potts 모델을 통해  
LaMnO₃의 distortion transition를 분석하였다. J2/J1 비율에 따른 상전이 온도 변화, 엔트로피 변화등을 분석하였으며 해당 논문의 자세한 내용은 아래 링크에서 확인할 수 있다.  <br>
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.014420  <br>


저는 이 논문을 참고하여, 동일한 현상이 2D Ising 모델로 단순화하였을 때에도 관찰될 수 있을지 궁금하여  
Anisotropic 2D Ising Model 분석을 진행하였다. 


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

### - Energy, Specific Heat Results

Energy, Specific Heat의 몬테카를로 시뮬레이션 코드는 `main.py`에 있다.
우리가 컨트롤하는 주요 parameters는 다음과 같다.




```python
Jx, Jy  = -1.0, 0.75

nt      = 100
N       = 16
eqSteps = 1000
mcSteps = 1200
nt      = 100
T       = np.linspace(1.5, 3.3, nt)
```
- Jx, Jy  : 교환 상수 J를 변화시키며 상전이 시점을 분석
- N  : 격자 한변의 길이
- eqSteps : 평형으로 가기 위한 Monte Carlo steps (시스템을 초기 랜덤 상태에서 안정화시키기 위해)
- mcSteps : 평형 이후 실제 측정에 사용하는 Monte Carlo steps
- nt : 시뮬레이션에서 샘플링할 개수
- T=np.linspace(1.5, 3.3, nt) : 1.5부터 3.3까지를 등간격으로 nt번 분할


Monte Carlo step(MCStep) 한번은 N × N번 스핀을 무작위로 선택해 에너지 변화를 계산하고 평균을 내는 과정이다.<br>
관측값 하나를 얻기 위해서는 지정한 mcSteps 횟수(1200)만큼 이 과정을 반복해 평균을 낸다.
이렇게 얻은 값을 온도 구간을 나눈 개수(nt)만큼 반복하면 시뮬레이션이 완료된다.  

nt, N, mcSteps 값을 늘릴수록 결과가 실제 해(정확한 값)에 가까워지지만, 그만큼 시뮬레이션 속도는 느려진다는 단점이 있다.
추가적으로, Jx < 0 , Jy > 0 로 설정하여 x축은 Anti-Ferromagnetic을 선호하고 y축은 Ferromagnetic을 선호하게 모델링하였다.  


여기서 분석하는 물리량은 에너지(E), 비열(C), 자화(M), 자기 감수율(X)이며 공식은 다음과 같다.  <br>  

```math
\mathcal{ E } = \frac{1}{N^2} \sum^{N^2}_{\langle i \rangle } H_i
```

```math
\mathcal{C} = \frac{\beta}{T} ( \langle  E^2 \rangle - \langle  E \rangle^2)
```

<br> 

- $N^2$은 무작위로 스핀을 뽑은 횟수
- $\beta= 1/k_b*T$ 이며 코드에서는 계산의 용이성을 위해 $( k_b=1 )$로 설정 -> `Beta = iT = 1.0/T[tt]`
<br>

비열 C의 공식에서 $( \langle E^2 \rangle - \langle E \rangle^2 )$는 에너지의 분산을 나타낸다. 따라서 C는 T에 따른 에너지의 분산임을 알 수 있다.  

따라서, mcstep을 반영한 에너지와 비열은 다음과 같다.

```math
\mathcal{\langle  E_{mc} \rangle} = \frac{1}{mcstep} \sum^{mcstep}_{\langle t=1 \rangle } E_t
```

```math
\mathcal{C_{mcstep}} = \frac{\beta}{T × {mcstep}^2} ( mcstep × \langle E^2_{mc} \rangle - \langle  E_{mc} \rangle^2)
```
```math
 \mathcal{\langle  E^2_{mc} \rangle} = \frac{1}{mcstep} \sum^{mcstep}_{\langle t=1 \rangle } E^2_t,\   \  {\langle  E_{mc} \rangle}^2 =  (\frac{1}{{mcstep}} \sum^{mcstep}_{\langle t=1 \rangle } E_t)^2 
```

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


### - Spin Ordering Across the Transition Temperature

다음으로, Transition Temperature를 기준으로 전, 후 스핀들의 배치가 어떻게 변하는 지 `spin ordering.py`를 통해 확인해보겠다. 
`spin ordering.py`의 mcmove는 `main.py`와 같다. 

```python
def output(f, config, i, n_, N):
    sp = f.add_subplot(2, 3, n_)
    plt.setp(sp.get_yticklabels(), visible=False)
    plt.setp(sp.get_xticklabels(), visible=False)

    color_map = np.zeros((N, N, 3))
    color_map[config == 1]  = [1.0, 0.5, 0.0]   # 주황색
    color_map[config == -1] = [0.0, 0.3, 0.8]   # 파란색

    plt.imshow(color_map, interpolation='nearest')
    plt.title(f'Monte Carlo step={i}')
    plt.axis('off')

    up_patch   = mpatches.Patch(color=[1.0, 0.5, 0.0], label='Spin Up')
    down_patch = mpatches.Patch(color=[0.0, 0.3, 0.8], label='Spin Down')
    plt.legend(handles=[up_patch, down_patch], loc="upper right", fontsize=8)
```
`sp = f.add_subplot(2, 3, n_)` : f의 subplot을 2행 3열로 출력, n_는 몇번 째에 위치할지 정함  
`plt.setp(sp.get_yticklabels(), visible=False)` : y축의 눈금을 제거  
`color_map = np.zeros((N, N, 3))` : N × N 를 0으로 초기화하고 3(R,G,B)으로 표시  
`plt.imshow(color_map, interpolation='nearest')` : color_map 배열을 픽셀 단위로 확대하여 색을 섞지 않고 격자 그대로 표시  

이제, $( J_x = -1 , J_y = 0.75 )$ 으로 설정하고 T를 변경해가며 스핀 정렬을 확인해보겠다.



#### 1) T < Transition Temperature( T = 0.4, )

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/814a9e04-7926-4e0d-bcc2-49bc25818aed" />

- 무작위로 정렬되어 있던 스핀들이 Monte Carlo step이 증가할 수록 세로 스트라이프 형태로 정렬됨을 확인할 수 있다.  
- $( J_x = -1 , J_y = 0.75 )$로 설정했기에, x축은 Anti Ferromagnetic, y축은 Ferromagnetic을 선호하여 세로 스트라이프 형태로 정렬된다
<br>
  <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/ed9b1eb6-3dde-4ef6-b893-6145ec191add" /> <br>

  
  
 
<br>

#### 2) T > Transition Temperature( T = 2.4)  

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/0d4c353b-4054-479b-97b9-6840a96ecb21" />

- Transition Temperature 이상에서 위와 동일한 step으로 시뮬레이션을 돌렸으나 스핀들이 정렬되지 않음을 확인할 수 있다.
- 이는 상전이가 일어나 스핀들이 무질서하게 배치되었음을 의미한다.
  


## Anisotropic 2D Ising Model with Diagonal Interactions

최근접 이웃 4개와 다음 근접 이웃인 대각선 이웃4개를 포함하여 몬테카를로 시뮬레이션을 진행하고, Jd 크기에 따라 평형상태가 어떻게 바뀌는지 분석했다.

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/8c1a71df-cc40-494d-ada7-0023a55f9318" />  


대각선 Jd를 반영한 에너지식은 다음과 같다. 


```math

\mathcal{E_{anisotropic}} = −J_x⋅S_{ab}(S_{a−1,b}	+S_{a+1,b}) - J_y⋅S_{ab}(S_{a,b+1} +S_{a,b−1})- J_d⋅S_{ab}(S_{a+1,b+1} +S_{a+1,b−1} +S_{a-1,b+1} +S_{a-1,b−1}) 

```
```math

\mathcal{ΔE_{anisotropic}} = 2⋅S_{ab}( J_x (S_{a−1,b}	+S_{a+1,b}) + J_y (S_{a,b+1} +S_{a,b−1}) + J_d (S_{a+1,b+1} +S_{a+1,b−1} +S_{a-1,b+1} +S_{a-1,b−1}) ) 

```

### - |Jx| = Jy ≠ Jd

  #### 1) Jd < |Jx| = Jy



