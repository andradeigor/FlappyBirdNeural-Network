# Flappy Bird Neural Netword

Projeto desenvolvido por [Igor Andrade](https://github.com/andradeigor). Este projeto foi feito como projeto final da disciplina Computação Científica e Análise de Dados, consiste na implementação de uma rede neural treinada via algoritmo genético para jogar Flappy Bird.

- [Teoria](#🖋-teoria)
- [Implementação](#💻-implementação)
- [Como Usar](#🤖-como-usar)
- [Demonstração](#📜-demonstração)
- [Tecnologias](#💻-tecnologias)
- [Contribuidores](#👥-contribuidores)
- [Licença](#📖-licença)

## 🖋 Teoria:

## 🧠 Redes Neurais

Para entender esse projeto precisamos primeiro ter em mente as partes que o compõem. Uma parte importante do código é feita via um algoritmo de Template Matching, usando o biblioteca OpenCV2. Esse algoritmo foi implementado e explicado por mim no trabalho final da disciplina [Álgebra Linear Algorítmica](https://github.com/andradeigor/CosineMatcher) e está sendo usado nesse projeto para detectar e localizar o pássaro e os canos do jogo Flappy Bird.

Com essa informação em mãos, passamos esses dados para uma Rede Neural que foi implementada do zero, seguindo o seguinte modelo:

![image-2](https://github.com/andradeigor/CosineMatcher/assets/21049910/5b077f70-f4fa-4e78-982f-826e0ca14a62)

Dentro de uma rede neural, nós temos esse nós de entrada que passam a informação que será analizada. No nosso caso, os dados de entrada são a localização do pássaro e dos canos. Após esses dados chegarem na camada de entrada eles são processados pelas camadas internas (hidden layers) e chegam no neurônio de saída da rede neural, que nos dá um valor entre [0-1], dependendo do valor que nós foi dado, nós realizamos ou não uma ação dentro do jogo.

Na realidade, o que está acontecendo na rede neural é que todos os neurônios das camadas anteriores são conectados aos neurônios da camada seguinte e são usados para gerar seus valores. Para melhor vizualização, tome como exemplo o neurônio $n_{11}$ da imagem acima, ele possui uma ligação com $\{a_1,a_2,a_3\}$ por meio das conexões $\{w_1,w_2,w_3\}$. Assim, o que ocorre matematicamente falando é que o valor de $n_{11}$ é calculado da seguinte maneira:

$$n_{11} = a_1.w_1+a_2.w_2+a_3.w3$$

Além disso, outro detalhe associado à cada neurônio é que cada um deles possuem um valor especial associado chamado de bias, que é somado ao final de todo cálculo feito. Assim, completando o exemplo acima temos:

$$n_{11} = a_1.w_1+a_2.w_2+a_3.w3+b_{11}$$

Esse valor especial em conjunto com os $w_i$ servem para determinar qual será o resultado de cada neurônio dependendo de cada entrada. Indo mais a fundo matematicamente falando, os $w_i$ são os pesos que cada entrada recebe no valor final, e o bias é um valor que desloca o resultado gerado por essa conta.

Após esse cálculo ser realizado para um neurônio, ao invés de pegarmos esse resultado e passar para frente, é adicionaddo mais um tratamento para esse valor gerado: uma função de ativação é usada nesse valor, e o resultado dessa função é passado como entrada para o próximo neurônio, e assim os valores vão caminhando pela rede neural.

![image](https://github.com/andradeigor/CosineMatcher/assets/21049910/4945954c-6b02-4fa4-acd7-160ad5be0da3)

Acima, temos um exemplo da função de ativação usada nesse projeto $y=tanh(x)$. Perceba que ao usarmos uma função de ativação, nós padronizamos os possíveis resultados de cada neurônio, não permitindo que eles cresçam indefinidamente, sempre nos gerando um resultado entre [-1,1]. Além disso, ao usarmos uma função de ativação ganha-se o poder de modelar sistemas não lineares dentro de uma rede neural, já que agora o cálculo da rede inteira não mais pode ser definido como um somátório de funções lineares com pesos.

Com isso, finalmente podemos descrever realmente o cálculo que ocorre em cada um dos neurônios:

$$n_{11} = tanh(a_1.w_1+a_2.w_2+a_3.w3+b_{11})$$

Generalizando o cálculo acima, temos:

$$n_{ij} = act((\sum_{k=1}^N a_{k}.w_{kj}) +  b_{ij})$$

Onde $N$ é o número de neurônios na camada anterior, $w_{kj}$ são os pesos associados aos neurônios anteriores $a_k$, $b_{ij}$ é o bias asssociado com o neurônio $n_{ij}$ e act é a função de ativação escolhida.

## 🧬 Algoritmo Genético

Além de redes neurais, o projeto também trata da implementação de um algoritmo genético para o treinamento, diferente da abordagem padrão de se usar Backpropagation. Esse algoritmo se baseia no processo de seleção natural.

- O processo se dá em um ciclo, no qual inicialmente geramos aleatoriamente uma série de "indivíduos" (no nosso caso, redes neurais). Esses indivíduos são avaliados em um critério heurístico feito para o problema em questão e no final eles recebem uma nota baseada no quão bem eles foram.

- Como esses indivíduos foram gerados aleatoriamente, dificilmente eles vão se sair bem de primeira. Mas, alguns vão, por sorte, se sair melhor que os outros. Esses indíviduos que tiveram a melhor pontuação serão escolhidos para servirem como "base" para a nova população.

- Após selecionados, passamos para a etapa de reprodução, na qual o "DNA" dos selecionados é usado de base para criar a nova população, pegando um pedaço de cada um deles de forma aleatória e compondo os novos. Ao final desse processo, visando fugir de mínimos locais, é também adicionada mutações aos DNA's criados, com uma chance bem baixa de ocorrer.

Por fim, essa população nova é avaliada e o ciclo se repete, como ilustra bem a imagem abaixo:

![genetic](https://github.com/andradeigor/CosineMatcher/assets/21049910/32eacce2-bd98-4760-81f0-d3ca67a0e1ad)

## 💻 Implementação:

Nosso objetivo é implementar uma rede neural que é treinada com base no algoritmo genético, o inicio do projeto foi implementar as operações básicas da rede neural. Para atingir esse objetivo, começou-se implementando um objeto inicial chamado de Layer que representa uma camada inteira da rede neural, com $n$ neurônios e realiza a operação de passar os dados por por eles.

Para isso, representamos essa camada como uma matriz $w_{nxm}$, onde $n$ é o número de neurônios e $m$ é o número de pesos que cada neurônio possui, que coincide com o número de neurônios presentes na camada anterior.

Assim, dado um vetor $i_{1xm}$ : $[i_1,i_2,...,i_m]$ que contêm o input dessa camada, podemos realizar os cálculos dessa camada como:

$$resultado = i_{1xm}.w_{nxm}^T = 
[\sum_{j=1}^m a_{1}.w_{1j},\sum_{j=1}^m a_{2}.w_{2j},...,\sum_{j=1}^m a_{m}.w_{mj}]$$

Agora, para que essa operação resulte nos cálculos que cada camada precisa fazer, basta somarmos um vetor $b_{1xm}$ que possui as biases de cada neurônio:

$$resultado = i_{1xm}.w_{nxm}^T + b_{1xm} =
[\sum_{j=1}^m a_{1}.w_{1j} + b_{1},\sum_{j=1}^m a_{2}.w_{2j} + b_{2},...,\sum_{j=1}^m a_{m}.w_{mj} + b_{m}]$$

Esse método foi implementado no objeto chamado de Layer da seguinte forma:

```python
def foward(self, input):
        self.output = np.dot(input,self.weights.T) + self.biases
```

Junto à método também foram adicionados um método de construção que gera valores aleatórios para os pesos, bem como uma função de ativação para gerar o resultado do cálculos. Ficando assim com:

```python
class Layer:
    def __init__(self,nInput=None,nNeurons=None):
        if(nInput==None or nNeurons==None): return
        self.weights = np.array(0.2 * np.random.randn(nNeurons,nInput))
        self.biases = np.array(0.2 * np.random.randn(1,nNeurons))
    def foward(self, input):
        self.output = np.dot(input,self.weights.T) + self.biases

    def tanh(self,values):
        self.result = np.tanh(values)

```

Com basse nessa classe e uma outra classe de Rede Neural que é capaz de criar várias camadas e fazer o cálculo para cada uma delas, foi possível implementar completamente a rede neural.

## 📜 Demonstração:

## 💻 Tecnologias

- Python
- Numpy
- OpenCV
- Pygame
- mss
- pynput

## 👥 Contribuidores

Esses são os contribuidores do projeto (<a href="https://allcontributors.org/docs/en/emoji-key">emoji key</a>).

<table>
  <tr>
    <td align="center"><a href="https://github.com/andradeigor"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/21049910?v=4" width="100px;" alt=""/><br /><sub><b>Igor Andrade</b></sub></a><br /><a href="https://github.com/andradeigor/DiscordBotUFRJ/commits?author=andradeigor" title="Igor Andrade">🤔 💻 🚧</a></td>
  </tr>
</table>

## 📖 Licença

Este projeto está licenciado sob a licença <a href="https://choosealicense.com/licenses/mit/">MIT</a>.
