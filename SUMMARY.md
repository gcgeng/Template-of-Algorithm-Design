#一轮省选前模板总结
##1.数据结构
###1.1单调队列
```cpp
//bzoj1010
for(int i = 1; i <= n; i++) {
        while(size >= 2) {
            int a = que[head];
            int b = que[head+1];
            if(calck(a, b) < g[i]) {
                head++;
                size--;
            }
            else break;
        }   
        int n = que[head];
        f[i] = f[n]+pow(s[i]-s[n]+i-n-1-l);
        if(size>=2){int x = que[tail];
        int y = que[tail-1];
        while(calck(x, i) < calck(y, x)) {
            tail--;
            size--;
            if(size < 2) break;
            x = que[tail];
            y = que[tail-1];
        }}
        tail++;
        que[tail] = i;
        size++;
    } 
```
###1.2字符串哈希
为后缀计算一个哈希值，满足$H(i)=H(i+1)x+s[i]$（其中$0 \leq i < n, H(n) = 0$），例如
$$H(4)=s[4]$$
$$H(3)=s[4]x+s[3]$$
一般地，$H(i)=s[n-1]x^{n-1-i}+s[n-2]x^{n-2-i}+...+s[i+1]x+s[i]$
对于一段长度为L的子串s[i]~s[i+L-1]，定义他的哈希值$Hash(i, L) = H(i)-H(i+L)x^L$。
预处理H和$x^L$。注意到hash值很大，我们把他存在unsigned long long中，这样就实现了自然溢出，可以大大减小常数。
###1.3树状数组
####1.3.1普通树状数组
树状数组好写好调，能用树状数组的时候尽量要使用。
**树状数组从1开始。**
```cpp
int lowbit(int x) { return x & -x; }
int sum(int x) {
  int ans = 0;
  while (x) {
    ans += bit[x];
    x -= lowbit(x);
  }
  return ans;
}
void add(int i, int x) {
  while (i <= n) {
    bit[i] += x;
    i += lowbit(i);
  }
}
```
####1.3.2支援区间修改的树状数组
假设对于区间$[a, b]$增加k，令g[i]为加了k以后的数组。
$$g[i] = s[i] (i < a)$$
$$g[i] = s[i] + i*k - k(a-1) (i >= a 且 i <= b)$$
$$g[i] = s[i] + b*k - k(a-1) (i > b)$$
所以我们开设两个树状数组。
####1.3.3二维树状数组
直接扩展就可以了。非常的直观和显然。
```cpp
//bzoj3132
void change(int id, int x, int y, int val) {
  for (int i = x; i <= n; i += i & -i) {
    for (int j = y; j <= m; j += j & -j) {
      c[id][i][j] += val;
    }
  }
}
int qu(int id, int x, int y) {
  int ans = 0;
  for (int i = x; i > 0; i -= i & -i) {
    for (int j = y; j > 0; j -= j & -j) {
      ans += c[id][i][j];
    }
  }
  return ans;
}
```
###1.4线段树
####1.4.1普通线段树
这个**线段树**版本来自bzoj1798，代表了一种最基础的线段树类型，支持区间修改，多重标记下传等等操作。

```cpp
//bzoj1798
void build(int k, int l, int r) {
  t[k].l = l;
  t[k].r = r;
  if (r == l) {
    t[k].tag = 1;
    t[k].add = 0;
    scanf("%lld", &t[k].sum);
    t[k].sum %= p;
    return;
  }
  int mid = (l + r) >> 1;
  build(k << 1, l, mid);
  build(k << 1 | 1, mid + 1, r);
  t[k].sum = (t[k << 1].sum + t[k << 1 | 1].sum) % p;
  t[k].add = 0;
  t[k].tag = 1;
}
void pushdown(int k) {
  if (t[k].add == 0 && t[k].tag == 1)
    return;
  ll ad = t[k].add, tag = t[k].tag;
  ad %= p, tag %= p;
  int l = t[k].l, r = t[k].r, mid = (l + r) >> 1;
  t[k << 1].tag = (t[k << 1].tag * tag) % p;
  t[k << 1].add = ((t[k << 1].add * tag) % p + ad) % p;
  t[k << 1].sum = (((t[k << 1].sum * tag) % p + (ad * (mid - l + 1) % p)%p)%p) % p;
  t[k << 1 | 1].tag = (t[k << 1 | 1].tag * tag) % p;
  t[k << 1 | 1].add = ((t[k << 1 | 1].add * tag) % p + ad) % p;
  t[k << 1 | 1].sum = (((t[k << 1|1].sum * tag) % p + (ad * (r-mid) % p)%p)%p) % p;
  t[k].add = 0;
  t[k].tag = 1;
  return;
}
void update(int k) { t[k].sum = (t[k << 1].sum%p + t[k << 1 | 1].sum%p) % p; }
void add(int k, int x, int y, ll val) {
  int l = t[k].l, r = t[k].r, mid = (l + r) >> 1;
  if (x <= l && r <= y) {
    t[k].add = (t[k].add + val) % p;
    t[k].sum = (t[k].sum + (val * (r - l + 1) % p) % p) % p;
    return;
  }
  pushdown(k);
  if (x <= mid)
    add(k << 1, x, y, val);
  if (y > mid)
    add(k << 1 | 1, x, y, val);
  update(k);
}
void mul(int k, int x, int y, ll val) {
  int l = t[k].l, r = t[k].r, mid = (l + r) >> 1;
  if (x <= l && r <= y) {
    t[k].add = (t[k].add * val) % p;
    t[k].tag = (t[k].tag * val) % p;
    t[k].sum = (t[k].sum * val) % p;
    return;
  }
  pushdown(k);
  if (x <= mid)
    mul(k << 1, x, y, val);
  if (y > mid)
    mul(k << 1 | 1, x, y, val);
  update(k);
}
ll query(int k, int x, int y) {
  int l = t[k].l, r = t[k].r, mid = (l + r) >> 1;
  if (x <= l && r <= y) {
    return t[k].sum%p;
  }
  pushdown(k);
  ll ans = 0;
  if (x <= mid)
    ans = (ans + query(k << 1, x, y)) % p;
  if (y > mid)
    ans = (ans + query(k << 1 | 1, x, y)) % p;
  update(k);
  return ans%p;
}
```
####1.4.2可持久化线段树
一种可持久化数据结构。
继承一样的节点，只是把修改的重新链接，节省空间。
容易爆空间。
经常与权值线段树和二分答案结合。
```cpp
int rt[maxn], lc[maxm], rc[maxm], sum[maxm];
void update(int l, int r, int x, int &y, int v) {
  y = ++sz;
  sum[y] = sum[x] + 1;
  if (l == r)
    return;
  lc[y] = lc[x];
  rc[y] = rc[x];
  int mid = (l + r) >> 1;
  if (v <= mid)
    update(l, mid, lc[x], lc[y], v);
  else
    update(mid + 1, r, rc[x], rc[y], v);
}
```
####1.4.3标记永久化
一种奇怪的姿势，又称李超线段树。
给节点打下的标记不进行下传，而是仅仅在需要的时候进行下传，这就是所谓永久化标记。
![](http://images2015.cnblogs.com/blog/890886/201703/890886-20170302072938266-693881259.png)
```cpp
struct line {
  double k, b;
  int id;
  double getf(int x) { return k * x + b; };
};
bool cmp(line a, line b, int x) {
  if (!a.id)
    return 1;
  return a.getf(x) != b.getf(x) ? a.getf(x) < b.getf(x) : a.id < b.id;
}
const int maxn = 50010;
line t[maxn << 2];
line query(int k, int l, int r, int x) {
  if (l == r)
    return t[k];
  int mid = (l + r) >> 1;
  line tmp;
  if (x <= mid)
    tmp = query(k << 1, l, mid, x);
  else
    tmp = query(k << 1 | 1, mid + 1, r, x);
  return cmp(t[k], tmp, x) ? tmp : t[k];
}
void insert(int k, int l, int r, line x) {
  if (!t[k].id)
    t[k] = x;
  if (cmp(t[k], x, l))
    std::swap(t[k], x);
  if (l == r || t[k].k == x.k)
    return;
  int mid = (l + r) >> 1;
  double X = (t[k].b - x.b) / (x.k - t[k].k);
  if (X < l || X > r)
    return;
  if (X <= mid)
    insert(k << 1, l, mid, t[k]), t[k] = x;
  else
    insert(k << 1 | 1, mid + 1, r, x);
}
void Insert(int k, int l, int r, int x, int y, line v) {
  if (x <= l && r <= y) {
    insert(k, l, r, v);
    return;
  }
  int mid = (l + r) >> 1;
  if (x <= mid)
    Insert(k << 1, l, mid, x, y, v);
  if (y > mid)
    Insert(k << 1 | 1, mid + 1, r, x, y, v);
}
```
### 1.5 平衡树

#### 1.5.1 Splay伸展树

一种最为常用的BBST。

## 2.图论

#### 2.1图的连通性

#### 2.2最短路与最小生成树
#### 2.3网络流
#### 2.4树相关
#### 2.5常用结论
#####2.5.1矩阵树(Matrix-Tree)定理



## 3.数学知识

### 3.1数论

#### 3.1.1扩展欧几里德算法

首先我们有欧几里德算法：

$$gcd(a, b) = gcd(a\  mod\ b, b)$$

扩展欧几里德算法解决了这样的问题：

$$ ax + by = gcd(a,b)$$

我们先考察一种特殊的情况：

当$b=0$时，我们直接可以有解：
$$
\begin{eqnarray}
\left\{
\begin{array}{lll}
x = 1 \\
y = 0
\end{array}
\right.
\end{eqnarray}
$$
一般地，我们令$c = a\ mod \ b$，递归地解下面的方程：

$$bx^{'}+cy^{'}=gcd(b,c)$$ 

根据欧几里德算法，有

$$bx^{'}+cy^{'}=gcd(a,b)$$

根据$mod$的定义我们可以有

$$c = a - b\lfloor\frac{a}{b}\rfloor$$

带入原式

$$bx^{'}+(a - b\lfloor\frac{a}{b}\rfloor)y^{'}=gcd(a,b)$$

为了体现与$a,b$的关系

$$ay^{'}+b(x^{'}-\lfloor\frac{a}{b}\rfloor y^{'})=gcd(a,b)$$

所以这样就完成了回溯。

这个算法的思想体现在了下面的程序里。

```c++
void gcd(int a, int b, int &d, int &x, int &y) {
  if(!b) {d = a; x = 1; y = 0; }
  else { gcd(b, a%b, d, y, x); y -= x * (a/b); }
}
```

#### 3.1.2线性筛与积性函数

##### 3.1.2.1线性筛素数

首先给出线性筛素数的程序。

```c++
void get_su(int n) {
  tot = 0;
  for(int i = 2; i <= n; i++) {
    if(!check[i]) prime[tot++] = i;
    for(int j = 0; j < tot; j++) {
      if(i * prime[j] > n) break;
      check[i * prime[j]] = true;
      if(i % prime[j] == 0) break;
    }
  }
}
```

可以证明的是，每个合数都仅仅会被他的最小质因数筛去，这段代码的时间复杂度是$\Theta (n)$的，也就是所谓线性筛。

> 证明：设合数$n$最小的质因数为$p$，它的另一个大于$p$的质因数为$p^{'}$，另$n = pm=p^{'}m^{'}$。 观察上面的程序片段，可以发现$j$循环到质因数$p$时合数n第一次被标记（若循环到$p$之前已经跳出循环，说明$n$有更小的质因数），若也被$p^{'}$标记，则是在这之前（因为$m^{'}<m$），考虑$i$循环到$m^{'}$，注意到$n=pm=p^{'}m^{'}$且$p,p^{'}$为不同的质数，因此$p|m^{'}$，所以当j循环到质数p后结束，不会循环到$p^{'}$，这就说明$n$不会被$p^{'}$筛去。

##### 3.1.2.2积性函数

* 考虑一个定义域为$N^{+}$的函数$f$（数论函数），对于任意两个**互质**的正整数$a,b$，均满足

$$f(ab) = f(a)f(b)$$，则函数*f*被称为积性函数。

* 如果对于任意两个正整数$a,b$，都有$f(ab)=f(a)f(b)$，那么就被称作完全积性函数。

容易看出，对于任意积性函数，$f(1)=1$。

* 考虑一个大于1的正整数$N$，设$N = \prod p_{i}^{a_i}$$，那么

$$f(N)=f(\prod p_i^{a_i})=\prod f(p_i^{a_i})$$，如果$f$还满足完全积性，那么

$$f(N)=\prod f(p_i)^{a_i}$$

* 如果$f$是一个任意的函数，它使得和式$g(m) = \sum_{d|m}f(d)$为积性函数，那么$f$也一定是积性函数。
* 积性函数的Dirichlet前缀和也是积性函数。这个定理是上面定理的反命题。
* 两个积性函数的Dirichlet卷积也是积性函数。

##### 3.1.2.3欧拉函数$\varphi$

* $\varphi(n)$表示$1..n$中与$n$互质的整数个数。
* 我们有欧拉定理：

$$n^{\varphi(m)}\equiv 1(mod\ m)\ \ \ \ n\perp m$$

可以使用这个定理计算逆元。

* 如果$m$是一个素数幂，则容易计算$\varphi(m)$，因为有$n \perp p^{k} \Leftrightarrow p \nmid n $ 。在$\{0,1,...,p^k-1\}$中的$p$的倍数是$\{0, p, 2p, ..., p^k-p\}$，从而有$p^{k-1}$个，剩下的计入$<span class="md-search-hit">\varphi</span>(p^k)$

$$\varphi(p^k) = p^k-p^{k-1}=(p-1)p^{k-1}$$

* 由上面的推导我们不难得出欧拉函数一般的表示：

$$\varphi(m) = \prod_{p|m}(p^{m_p}-p^{m_p-1}) = m \prod_{p|m}(1-\frac{1}{p})=\prod(p-1)p^{m_p-1}$$

* 运用Mobius反演，不难得出$\sum_{d|n}\varphi(d) = n$。
* 当$n>1$时，$1..n$中与$n$互质的整数和为$\frac{n\varphi(n)}{2}$
* 降幂大法$$A^B\ mod\ C=A^{B\ mod\ \varphi(C)+\varphi(C)}\ mod\ C$$

##### 3.1.2.4线性筛法求解积性函数

* 积性函数的关键是如何求$f(p^k)$。
* 观察线性筛法中的步骤，筛掉n的同时还得到了他的最小的质因数$p$，我们希望能够知道$p$在$n$中的次数，这样就能够利用$f(n)=f(p^k)f(\frac{n}{p^k})$求出$f(n)$。
* 令$n=pm$，由于$p$是$n$的最小质因数，若$p^2|n$，则$p|m$，并且$p$也是$m$的最小质因数。这样在筛法的同时，记录每个合数最小质因数的次数，就能算出新筛去合数最小质因数的次数。
* 但是这样还不够，我们还要能够快速求解$f(p^k)$，这时一般就要结合$f$函数的性质来考虑。
* 例如欧拉函数$\varphi$，$\varphi(p^k)=(p-1)p^{k-1}$，因此进行筛法求$\varphi(p*m)$时，如果$p|m$，那么$p*m$中$p$的次数不为1,所以我们可以从$m$中分解出$p$，那么$\varphi(p*m) = \varphi(m) * p$，否则$\varphi(p * m) =\varphi(m) * (p-1)$。


* 再例如默比乌斯函数$\mu​$，只有当$k=1​$时$\mu(p^k)=-1​$，否则$\mu(p^k)=0​$，和欧拉函数一样根据$m​$是否被$p​$整除判断。

```c++
void get_phi(int n) {
  memset(check, 0, sizeof(check));
  phi[1] = 1;
  int tot = 0;
  for(int i = 2; i <= n; i++) {
    if(!check[i]) {
      prime[tot++] = i;
      phi[i] = i-1;
    }
    for(int j = 0; j < tot; j++) {
      if(i * prime[j] > n) break;
      check[i*prime[j]]=true;
      if(i % prime[j] == 0) {
        phi[i * prime[j]] = phi[i] * prime[j];
        break;
      } else {
        phi[i * prime[j]] = phi[i] * (prime[j]-1);
      }
    }
  }
}
```

```c++
void get_mu(int n) {
  memset(check, 0, sizeof(check));
  mu[1] = 1;
  int tot = 0;
  for(int i = 2; i <= n; i++) {
    if(!check[i]) {
      prime[tot++] = i;
      mu[i] = -1;
    }
    for(int j = 0; j < tot; j++) {
      if(i * prime[j] > n) break;
      check[i * prime[j]] = true;
      if(i % prime[j] == 0) {
        mu[i * prime[j]] = 0;
        break;
      } else {
        mu[i * prime[j]] = -mu[i];
      }
    }
  }
}
```

##### 3.1.2.5线性筛逆元

令$f(i)$为$i$在$mod\ p$意义下的逆元。显然这个函数是积性函数，我们可以使用线性筛求。但是其实没有那么麻烦。

我们设$p = ki+r$，那么$ki+r \equiv 0 (mod\ p)$，两边同时乘$i^{-1}r^{-1}$，有$kr^{-1}+i^{-1}\equiv 0$，那么$i^{-1} \equiv -kr^{-1}=-\lfloor \frac {p}{i} \rfloor * (p \ mod\ i)^{-1}$，这样就可以递推了。

```c++
void getinv(int n) {
  inv[1] = 1;
  for(int i = 2; i <= x; i++)
    inv[i] = (long long)(p - p/i)*inv[p % i] % p;
}
```

有了逆元，我们就可以预处理阶乘的逆元

$$n!^{-1} \equiv \prod_{k=1}^n k^{-1}\ mod \ p$$



#### 3.1.3默比乌斯反演与狄利克雷卷积

##### 3.1.3.1初等积性函数$\mu$

$\mu$就是容斥系数。
$$
\mu(n)=\begin{eqnarray}
\left  \{
\begin{array}{lll}
0 , \exists x^2|n\\
(-1)^k,n=\prod_{i=1}^{k}p_i
\end{array}
\right.
\end{eqnarray}
$$
$\mu$函数也是一个积性函数。

下面的公式可以从容斥的角度理解。
$$
\sum_{d|n}\mu(d)=[n=1]
$$

##### 3.1.3.2默比乌斯反演

首先给出Mobius反演的公式：

$$
F(n)=\sum_{d|n}f(d) \rightarrow f(n)=\sum_{d|n}\mu(\frac{n}{d})F(d)
$$
有两种常见的证明，一种是运用Dirichlet卷积，一种是使用初等方法。

证明：
$$
\sum_{d|n}\mu(d)F(\frac{n}{d}) = \sum_{d|n}\mu(\frac{n}{d})F(d)=\sum_{d|n}\mu(\frac{n}{d})\sum_{k|d}f(k)\\=\sum_{d|n}\sum_{k|d}\mu(\frac{n}{d})f(k)=\sum_{k|n}\sum_{d|\frac{n}{k}}\mu(\frac{n}{kd})f(k)\\
=\sum_{k|n}\sum_{d|\frac{n}{k}}\mu(d)f(k)=\sum_{k|n}[\frac{n}{k} = 1]f(k)=f(n)
$$
默比乌斯反演的另一种形式：
$$
F(n)=\sum_{n|d}f(d)\rightarrow f(n)=\sum_{n|d}\mu(\frac{d}{n})F(d)
$$
这个式子的证明与上式大同小异，我在这里写一下关键步骤
$$
\sum_{n|d}\sum_{d|k}\mu(\frac{d}{n})f(k)=\sum_{n|k}\sum_{d|\frac{k}{n}}\mu(d)f(k)=f(n)
$$
对于一些函数$f(n)$，如果我们很难直接求出他的值，而容易求出倍数和或者因数和$F(n)$，那么我们可以通过默比乌斯反演来求得$f(n)$的值

##### 3.1.3.3狄利克雷卷积

数论函数$f$和$g$的狄利克雷卷积定义为$(f*g)(n) = \sum_{d|n}f(d)g(\frac{n}{d})$，狄利克雷卷积满足交换律，结合率，对于加法满足分配律，存在单位元$e(n)=[n=1]$。

#### 3.1.4积性函数求和与杜教筛

##### 3.1.4.1默比乌斯函数求前缀和

求$\sum_{i=1}^{n}\mu(i)$





### 3.2组合数学

### 3.3常见结论与技巧

#### 3.3.1裴蜀定理

若a,b是整数,且（a,b)=d，那么对于任意的整数x,y,ax+by都一定是d的倍数，特别地，一定存在整数x,y，使ax+by=d成立。

它的一个重要推论是：a,b互质的充要条件是存在整数x,y使ax+by=1.

#### 3.3.2底和顶

* 若连续且单调增的函数$f(x)$满足当$f(x)$为整数时可推出$x$为整数，则$$\lfloor f(x) \rfloor = \lfloor f(\lfloor x \rfloor) \rfloor$$和$\lceil f(x) \rceil = \lceil f(\lceil x\rceil) \rceil$
* $$\lfloor \frac {\lfloor\frac{x}{a} \rfloor}{b}\rfloor = \lfloor \frac{x}{ab}\rfloor$$
* 对于$i$，$\lfloor \frac{n}{\lfloor \frac{n}{i}\rfloor}\rfloor$是与$i$被$n$除并下取整取值相同的一段区间的右端点

#### 3.3.3和式

##### 3.3.3.1三大定律

* 分配律 $$\sum_{k \in K} c a_k = c \sum_{k \in K} a_k$$
* 结合律$$\sum_{k \in K}(a_k + b_k)=\sum_{k \in K}a_k+\sum_{k \in K}b_k$$
* 交换律$$\sum_{k \in K}a_k=\sum_{p(k) \in K} a_{p(k)}$$其中p(k)是n的一个排列
* 松弛的交换律:若对于每一个整数$n$，都恰好存在一个整数$k$使得$p(k)=n$，那么交换律同样成立。
##### 3.3.3.2求解技巧

* 扰动法，用于计算一个和式，其思想是从一个未知的和式开始，并记他为$S_n$：$$S_n=\sum_{0 \leq k \leq n} a_k$$，然后，通过将他的最后一项和第一项分离出来，用两种方法重写$S_{n+1}$，这样我们就得到了一个关于$S_n$的方程，就可以得到其封闭形式了。

* 一个常见的交换
  $$
  \sum_{d|n}f(d)=\sum_{d|n}f(\frac{n}{d})
  $$





##### 3.3.3.3多重和式

* 交换求和次序：

$$
\sum_j\sum_ka_{j,k}[P(j,k)]=\sum_{P(j,k)}a_{j,k}=\sum_k\sum_ja_{j,k}[P(j,k)]
$$

* 一般分配律：$$\sum_{j \in J, k \in K}a_jb_k=(\sum_{j \in J}a_j)(\sum_{k \in K}b_k)$$

* $Rocky\ Road$
  $$
  \sum_{j \in J}\sum_{k \in K(j)}a_{j,k}=\sum_{k \in K^{'}}\sum_{j \in J^{'}}a_{j,k}
  $$








$$
[j \in J][k \in K(j)]=[k \in K^{'}][j \in J^{'}(k)]
$$

事实上，这样的因子分解总是可能的：我们可以设$J=K^{'}$是所有整数的集合，而$K(j)$和$J^{'}(K)$是与操控二重和式的性质$P(j,k)$相对应的集合。下面是一个特别有用的分解。

$$[1\leq j \leq n][j \leq k \leq n] = [1 \leq j \leq k \leq n] = [1 \leq k \leq n][1 \leq j \leq k]$$

* 一个常见的分解
  $$
  \sum_{d|n}\sum_{k|d}=\sum_{k|m}\sum_{d|\frac{m}{k}}
  $$







* 一个技巧

  如果我们有一个包含$k+f(j)$的二重和式，用$k-f(j)$替换$k$并对$j$求和比较好。

####3.3.4数论问题的求解技巧
* $\{\lfloor \frac{n}{i} \rfloor|i \in [1,n]\}$只有$O(\sqrt n)$种取值。所以可以使用这个结论降低复杂度。

例如，在bzoj2301中，我们最终解出了$$f(n, m)=\sum_{1 \leq d \leq min(n, m)}\mu(d)\lfloor \frac {n}{d} \rfloor \lfloor \frac {m}{d} \rfloor$$我们就可以使用杜教筛计算出默比乌斯函数的前缀和，计算出商与除以i相同的最多延伸到哪里，下一次直接跳过这一段就好了。下面是这个题的一段程序。

```c++
int calc(int n, int m) {
    int ret = 0, last;
    if(n > m) std::swap(n, m);
    for(int i = 1; i <= n; i = last + 1) { //i就相当于原式中的d
        last = min(n / (n/i), m / (m/i));  //last计算了商与除以i相同的最多延伸到哪里，不难证明这样计算的正确性
        ret += (n / i) * (m / i) * (sum[last] - sum[i-1]);
    }
    return ret;
}
```





### 3.4博弈论

### 3.5其他数学工具
#### 3.5.1快速乘

```c++
inline ll mul(ll a, ll b) {
  ll x = 0;
  while (b) {
    if (b & 1)
      x = (x + a) % p;
    a = (a << 1) % p;
    b >>= 1;
  }
  return x;
}
```

#### 3.5.2快速幂

```c++
inline ll pow(ll a, ll b, ll p) {
  ll x = 1;
  while (b) {
    if (b & 1)
      x = mul(x, a);
    a = mul(a, a);
    b >>= 1;
  }
  return x;
}
```
#### 3.5.3更相减损术

第一步：任意给定两个正整数；判断它们是否都是偶数。若是，则用2约简；若不是则执行第二步。

第二步：以较大的数减较小的数，接着把所得的差与较小的数比较，并以大数减小数。继续这个操作，直到所得的减数和差相等为止。

则第一步中约掉的若干个2与第二步中等数的乘积就是所求的最大公约数。



## 4.动态规划

## 5.其他重要工具

### 5.1位运算

### 5.2非递归 DFS

见[Menci的博客](https://oi.men.ci/non-recursion-dfs-with-stack/)




