---
layout: post
thumbnail: "assets/understanding-VAE/var-inf.png"
---

In this post I attempt to describe Variational Autoencoders (VAE) both from a theoretical and a practical point of view. The first paper to introduce VAE [Kingma et al. 2014] assumes the reader knows quite a bit of Variational Inference (VI). For those of us lacking this consept, i will take the same route the authors took in the original paper, but we will take a closer look on VI. Then, we will review the intersection of autoencoders and VI, the variational autoencoder.
But first, let's start with a really quick overview of vanilla autoencoders

# Autoencoders

Generally speaking, an autoencoder is an unsupervised method designed to reconstruct some input data $$\textbf{x} \in X$$. The main feature of the autoencoder is the ability to learn a concise representation of the features of $$\textbf{x}$$, <i>i.e.</i>, $$\textbf{z} \in Z$$, while being able to reconstruct $$\textbf{x}$$ from $$\textbf{z}$$. In order to create the reconstruction $$\textbf{x}'$$ the autoencoder uses two functions. The first, an encoder, denoted by $$q_\phi(\textbf{x} \mid \textbf{z}) $$, and a decoder, denoted by $$p_\theta(\textbf{z} \mid \textbf{x})$$, with parameters $$\phi,~\theta$$ respectively. Notice that $$\dim(\textbf{z}) < \dim(\textbf{x})$$ otherwise the model will not learn anything of use. We want to minimize the reconstruction loss using some loss function, for example the MSE: 

$$ (1) \qquad \phi, \theta = \arg\,min_{\phi, \theta} \| \textbf{x} - p_\theta(q_\phi(\textbf{x}))\|^2$$

From a coding theory perspective, the encoder takes the input, or observed variable, $$\textbf{x} \in X$$ and maps it, in our case, using a Multi Layer Perceptron (MLP), to a latent or hidden variable, or code, $$\textbf{z} \in Z$$. Then the decoder maps $$\textbf{z}$$, again, using an MLP, to the reconstruction, $$\textbf{x}'$$.

Throughout reading this post, lets think of $$X \sim p_{\theta^*}(\textbf{x})$$ as a distribution of digit drawings, like MNIST. So the autoencoder's goal is to get $$\textbf{x}$$, an image of a digit, "shrink" its representation to a smaller dimention $$\textbf{z}$$, then reconstructing $$\textbf{x}$$, while not losing (much) information.

# Method
## Problem Scenario

After reading the previous section, we might notice that autoencoders fit the latent variable model paradigm, in which we assume the observable data $$\textbf{x} \in X$$ is generated by some random process involving the continuous latent variables $$\textbf{z}$$. <i>i.e.</i>, We generate $$\textbf{z}$$ from the prior distribution $$p_{\theta^*}(\textbf{z})$$, then $$\textbf{x}$$ is generated from the conditional distribution $$p_{\theta^*}(\textbf{x} \mid \textbf{z})$$. Where the distributions $$p_{\theta}(\textbf{z})$$, $$p_{\theta}(\textbf{x} \mid \textbf{z})$$ depend on some parameters $$\mathbf{\theta}$$. Notice that we do not have any information regarding the real values of $$\textbf{z}$$ or the optimum value of parameters $$\theta^*$$.

(A note regarding notation. A more verbose notation could be: $$p_{\theta_1}(\textbf{z})$$, $$p_{\theta_2}(\textbf{x} \mid \textbf{z})$$ are parametric functions and the real values of $$\theta = (\theta_1, \theta_2)$$ are unknown. We will use the former notation for consistency with Kingma et al.'s paper) 

### Generation Process

In addition to the reconstruction goal defined in the autoencoders section, we would like to be able to execute a generation process: Assume you can generate $$\textbf{z} \sim p(\textbf{z})$$, generate $$\textbf{x}$$ from $$p_\theta(\textbf{x} \mid \textbf{z})$$. 

Why is this a generative process? 

Again, think of $$X$$ as a set of images. These images come from some distribution $$p_{\theta^*}(\textbf{x})$$ we know nothing about. But if we can find the conditional distribution $$p_{\theta}(\textbf{x} \mid \textbf{z})$$ (And we know $$p_{\theta}(\textbf{z})$$) we can generate images!

![](/assets/understanding-VAE/generator2.png)
<br>*The Generation Process*

Specifically, we would like to find $$\theta$$ such that will maximize $$p_\theta(\textbf{x})$$ for training images $$X$$, via the generation process $$p(\textbf{x} \mid \textbf{z})$$: Optimize parameters $$\theta$$ such that the generative process $$p_\theta(\textbf{x} \mid \textbf{z})$$ will return $$\textbf{x}'$$ that are as close to the $$\textbf{x} \in X$$.

$$
    \max_\theta p_\theta(\textbf{x}) = \max_\theta \int_z p(\textbf{z})p_\theta(\textbf{x} \mid \textbf{z})
$$


Great, to compute this integral we only need to go over all possible $$Z$$ and maximize $$p_\theta(\textbf{x} \mid \textbf{z})$$ - The thing is, this is intractable.

Think about a simpler case, assume $$\textbf{z}$$ is a boolean vector in $$d$$ dimensions, then there are $$2^d$$ possible configurations of $$\textbf{z}$$, which is exponential in $$d$$ and well too big to integrate over. Our case is much worse, $$\textbf{z}$$ is continuous.

To recap, we want to be able to generate data using $$p_\theta(\textbf{x} \mid \textbf{z})$$, we understand we can learn a generative process by maximizing  $$\max_\theta p_\theta(\textbf{x})$$, but this is intractable. Let's leave this for a bit, but don't forget this is our goal, and discuss another problem that might help us with this later.

# Sidetracking, Introduction to Variational and Posterior Inference

In VI, just as before, we think of the data distribution as the marginal distribution of this:

$$
    p(\textbf{x}) = \int_z p(\textbf{x}, \textbf{z}) = \int_z p(\textbf{z}) p(\textbf{x} \mid \textbf{z})
$$

In the previous section we wanted to infer about the likelihood $$p(\textbf{x} \mid \textbf{z})$$ but we hit a brick wall. Let's look at this from another point of view and try to infer about the posterior, $$p(\textbf{z} \mid \textbf{x})$$. In order to do this we will use VI, a method that approximates probability densities through optimization. The probability density we want to approximate is the posterior, the conditional distribution of the hidden variables given the observed variables.

$$
    p(\textbf{z} \mid \textbf{x}) = \frac{p(\textbf{z}, \textbf{x})}{p(\textbf{x})}
$$

The denominator, $$p(\textbf{x})p(x)$$, as we saw beforehand, is intractable. Variational Inference(VI) suggests to find another distribution that we know how to sample from and is close to the original posterior distribution, $$p(\textbf{z} \mid \textbf{x})p(z \sim x)$$.

![](/assets/understanding-VAE/var-inf.png)
*<br>Image from [this](https://www.youtube.com/watch?v=ogdv6dbvVQ) NIPS tutorial. Notice in this post we use $$\phi$$ instead of $$\nu$$ as the distribution's parameters.*


Formally, Variational Inference suggest that instead of trying to infer about something we don't know how to compute, lets try to optimize the proximity of a new distribution $$q_\phi(\textbf{z} \mid \textbf{x})$$ to the original posterior $$p(\textbf{z} \mid \textbf{x})$$. How does it work? Select a variational family of distributions over the latent variables we know how to sample from, denoted by $$Q_\phi(\textbf{z})$$. Choose some parametric distribution $$q_\phi(\textbf{z} \mid \textbf{x}) \in Q_\phi(\textbf{z})$$. Then, search for $$\phi$$ that approximates $$q_{\phi}(\textbf{z} \mid \textbf{x})$$ to $$p(\textbf{z} \mid \textbf{x})$$. Now we can approximating sampling from $$p(\textbf{z} \mid \textbf{x})$$ by sampling from $$q_{\phi}(\textbf{z} \mid \textbf{x})$$.

For example, assume we want to find an approximation to $$p(\textbf{z} \mid \textbf{x})$$ we hypothesize that it can be approximated by a normal distribution with some mean and variance. Denote $$Q_\phi(\textbf{z})$$ as the family of normal distributions $$\mathcal{N}(\phi_{\mu}, \phi_{\sigma^2})$$. We are looking for $$\phi^*$$ such that $$q_{\phi^*}(z)=\mathcal{N}(\phi_{\mu^*}, \phi_{\sigma^*})$$  is as close as possible to the real posterior, $$p(\textbf{z} \mid \textbf{x})$$.

![](/assets/understanding-VAE/var-inf2.png){:height="75%" width="75%"}
*<br>The plot shows the original distribution (yellow) along with the Laplace (red) and variational (green) approximations. Image from [11]*



Why is this easier then our original question? our goal now is optimizing and not inferring, and optimizing is something we do quite a lot, just use your favorite gradient decent algorithm.

(Another note regarding notation. In VI we indeed look for some distribution $$q_\phi(\textbf{z})$$ approximating $$p(\textbf{z} \mid \textbf{x})$$ that may or may not depend on input $$x$$. In Kingma et al.'s paper we also care about $$q_\theta(\cdot)$$ with respect to the input (Think about the reconstruction process). For this reason we specifically look for a distribution that depends on the input, so from now on, we note this as $$q_\phi(\textbf{z} \mid \textbf{x})$$)

So our optimization problem now, for the posterior inference, is minimizing the proximity of  $$q_\phi(\textbf{z} \mid \textbf{x})$$ and $$p(\textbf{z} \mid \textbf{x})$$. Usually (but not always) in VI, closeness is defined by the Kullback–Leibler divergence defined over distributions $$p(x)$$ and $$q(x)$$ by

$$
\begin{aligned}
D_{KL}(q \Vert p) = \int_x q(x) \log \frac{q(x)}{p(x)}
\end{aligned}
$$

Or for the minimization problem of our two distributions:

$$
\begin{aligned}
\min_\phi D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert p(\textbf{z} \mid \textbf{x})) &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} \mid \textbf{x})} dz
\end{aligned}
$$

The second input to the KL divergence is the same posterior distribution we want to find, so we can't compute it. Let's see if there's anything else we can say about this equation.

$$
\begin{aligned}
D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert p(\textbf{z} \mid \textbf{x})) &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} \mid \textbf{x})} dz \\
    &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x}) p(\textbf{x})}{p(\textbf{z} , \textbf{x})} dz \\
    &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) (\log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} , \textbf{x})} + \log p(\textbf{x})) dz \\
    &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} , \textbf{x})}dz + \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log p(\textbf{x}) dz\\
    &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} , \textbf{x})}dz + \log p(\textbf{x}) \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) dz\\
    &=  \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} , \textbf{x})}dz + \log p(\textbf{x})\\
\end{aligned}
$$

Define $$L(\phi; \textbf{x})$$ to be the negation of the first term:

$$
\begin{aligned}
L(\phi; \textbf{x}) = -\int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p(\textbf{z} , \textbf{x})}dz &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{p(\textbf{z}, \textbf{x})}{q_\phi(\textbf{z} \mid \textbf{x})}dz\\
&= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{p(\textbf{z})p(\textbf{x} \mid \textbf{z})}{q_\phi(\textbf{z} \mid \textbf{x})}dz
\end{aligned}
$$

And we get:

$$
\begin{aligned}
    \log p(\textbf{x}) &= D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert P(\textbf{z} \mid \textbf{x})) + L(\phi; \textbf{x}) \\
\end{aligned}
$$

Do you remember I said Kingma et al. assumes quite a bit about the readers knowledge in VI, this is (almost) equation 1 in their paper.


The first term in the RHS is always positive, so the second term is the lower bound of the equation, and by that of $$p(\textbf{x})$$, the evidence. For this reason $$L(\phi; \textbf{x})$$ is known as the Evidence Lower BOund or ELBO.


# Back on Track

Recall our original goal, we wanted to maximize $$p(\textbf{x})$$, but we couldn't write it in a tractable form. The above equation means instead of directly maximizing $$p(\textbf{x})$$, we can maximize its lower bound.

$$
\begin{aligned}
\max_\phi L(\phi; \textbf{x})
\end{aligned}
$$

We can now join both of our goals, learn $$\theta$$ for the generation process together with learning $$\phi$$ for $$p_\theta(\textbf{x} \mid \textbf{z})$$., the posterior .

$$
\begin{aligned}
\max_{\theta,\phi} L(\theta,\phi; \textbf{x}) &= \int_{z} q_\phi (\textbf{z} \mid \textbf{x}) \log \frac{p(\textbf{z})p_\theta(\textbf{x} \mid \textbf{z})}{q_\phi(\textbf{z} \mid \textbf{x})}dz\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [-\log q_\phi(\textbf{z} \mid \textbf{x}) +\log p_\theta(\textbf{x}, \textbf{z})]
\end{aligned}
$$

![](/assets/understanding-VAE/enc-dec.png)

$$L(\theta,\phi; \textbf{x})$$ is now expressed as an estimation. Using Monte Carlo estimate we can estimate it as its average: Generate $$L$$ different $$\textbf{z}$$ samples and average the inner function of $$L(\theta,\phi; \textbf{x})$$

$$
\begin{aligned}
\widetilde{L}(\theta,\phi; x) &= \frac{1}{L} \sum_{l=1}^L \log p_\theta(\textbf{x}, \textbf{z}^{(l)}) - \log q_\phi(\textbf{z}^{(l)} \mid \textbf{x})\\
\text{where}~ \textbf{z}^{(l)} &\sim q_\phi (\textbf{z} \mid \textbf{x})
\end{aligned}
$$

Finally we have a function we can optimize, recall gradient decent algorithms optimization step ($$w_{i+1} = w_i - \lambda\nabla L(w_i)$$ for some rate $$\lambda$$), we now need to compute the gradient of $$L(\theta,\phi; \textbf{x})$$ with respect to both $$\theta$$ and $$\phi$$.

The gradient with respect to $$\theta$$, $$\nabla_\theta$$,  is fairly straight forward because we can push the gradient inside the sum:

$$
\begin{aligned}
\nabla_\theta\widetilde{L}(\theta,\phi; x) =& \frac{1}{L} \sum_{l=1}^L \nabla_\theta\log q_\phi(\textbf{z}^{(l)} \mid \textbf{x})\\
\text{where}~& \textbf{z}^{(l)} \sim q_\phi (\textbf{z} \mid \textbf{x})
\end{aligned}
$$

Doing the same for $$\nabla_\phi$$ is not possible as we sample $$\textbf{z}^{(l)}$$ from $$q_\phi (\textbf{z} \mid \textbf{x})$$ which depends on $$\phi$$. While we can find a gradient estimate using the Log Derivative Trick, empirically, these gradients are too large. 

Why does it matter? In optimization, the variance of the gradient is more important than the value of the function itself. The gradient impacts the practicability and the speed of the convergence of the optimization problem, so this is not quite the solution we were looking for. (Interestingly though, the result of the gradient using the Log Derivative Trick is just like REINFORCE, sometimes also called the likelihood ratio)

If we can't push the gradient inside the sum because $$\textbf{z}^{(l)}$$ is sampled from $$q_\phi (\textbf{z} \mid \textbf{x})$$, can we sample $$\textbf{z}^{(l)}$$ independently of $$\phi$$ while still $$\textbf{z}^{(l)} \sim q_\phi (\textbf{z} \mid \textbf{x})$$? What Kingma et al. suggested as a solution, known as the reparameterization trick, is to sample $$\textbf{z} \sim q_\phi (\textbf{z} \mid \textbf{x})$$ using a new auxiliary parameter-free variable $$\epsilon \sim p(\epsilon)$$ ($$\epsilon$$ is a vector, having trouble bolding it), and a parametric function $$g_\phi(\cdot)$$, such that now $$\textbf{z}$$ is a deterministic variable $$\textbf{z} = g_\phi(\epsilon, \textbf{z})$$. This means, we can write our objective function now as: 

$$
\begin{aligned}
\widetilde{L}(\theta,\phi; x) &= \frac{1}{L} \sum_{l=1}^L \log p_\theta(\textbf{x}, \textbf{z}^{(l)}) - \log q_\phi(\textbf{z}^{(l)} \mid \textbf{x})\\
\text{where}~ \textbf{z}^{(l)}&=g_\phi(\epsilon^{(l)}, \textbf{x})~\text{and}~\epsilon^{(l)} \sim p (\epsilon)
\end{aligned}
$$

To put it in a practical example, assume $$\textbf{z} \sim q_\phi (\textbf{z} \mid \textbf{x}) = \mathcal{N}(\phi_\mu, \phi_\sigma^2)$$. The reparameterization trick suggests to define $$\epsilon \sim \mathcal{N}(\textbf{0},\textbf{I})$$ and $$g_\phi(\epsilon, \textbf{z}) = \mu + \sigma^2\odot\epsilon$$. Where $$\odot$$ signify an element-wise product. Now, $$\textbf{z}^{(l)}$$ can be written as $$g_\phi(\epsilon^{(l)}, \textbf{x})$$. Essentially, this reparameterization trick reduces the variance dramatically.

Another novelty the authors suggested in order to further reduce the variance of the gradient is to write $$L(\theta,\phi; \textbf{x})$$ differently.

$$
\begin{aligned}
    L(\theta,\phi; \textbf{x}) &= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log p_\theta(\textbf{x}, \textbf{z}) - \log q_\phi(\textbf{z} \mid \textbf{x})]\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log p_\theta(\textbf{x} \mid \textbf{z})p_\theta(\textbf{z})- \log q_\phi(\textbf{z} \mid \textbf{x})]\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log p_\theta(\textbf{x} \mid \textbf{z}) + \log p_\theta(\textbf{z})- \log q_\phi(\textbf{z} \mid \textbf{x})]\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log p_\theta(\textbf{x} \mid \textbf{z})] + \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log p_\theta(\textbf{z}) -\log q_\phi(\textbf{z} \mid \textbf{x})]\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log p_\theta(\textbf{x} \mid \textbf{z})] - \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [\log q_\phi(\textbf{z} \mid \textbf{x}) - \log p_\theta(\textbf{z})]\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [ \log p_{\theta}(\textbf{x} \mid \textbf{z})] - \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})}  [\log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p_{\theta}(\textbf{z})}]\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [ \log p_{\theta}(\textbf{x} \mid \textbf{z})] - \int_{z} q_\phi(\textbf{z} \mid \textbf{x}) \log \frac{q_\phi(\textbf{z} \mid \textbf{x})}{p_{\theta}(\textbf{z})}\\
&= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [ \log p_{\theta}(\textbf{x} \mid \textbf{z})] - D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert p_{\theta}(\textbf{z}))
\end{aligned}
$$

That is: 
$$
\begin{aligned}
L(\theta,\phi; \textbf{x}) &= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [ \log p_{\theta}(\textbf{x} \mid \textbf{z})] - D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert p_{\theta}(\textbf{z}))\\
\text{where}~ \textbf{z}^{(l)}&=g_\phi(\epsilon^{(l)}, \textbf{x})~\text{and}~\epsilon^{(l)} \sim p (\epsilon)
\end{aligned}
$$

Again, empirically, this variation shows less variance than the previous way. A motivation for this can be that unlike in the previous definition of $$L(\theta,\phi; \textbf{x})$$, here only the first term is an expectation and needs to be assessed using Monte Carlo estimate, the second term can be integrated analytically. 

# Variational Autoencoder

Alright, lets tie this all together! From the previous section we finalized our  objective function for both the encoder and the decoder (Previous equation). First lets try to understand it, recall Eq. (1), where we defined a loss function over the reconstruction process. In variational autoencoders we still care about this reconstruction, manifested in $$\mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [ \log p_{\theta}(\textbf{x} \mid \textbf{z})]$$. But we also want to regularize $$q_\phi (\textbf{z} \mid \textbf{x})$$ by keeping it close to the latent variable distribution, $$p_{\theta}(\textbf{z})$$.

![](/assets/understanding-VAE/var-enc-dec.png)
<br>*The Variational Autoencoder*

Now, an example:

Assume for this example, all distributions are Gaussian:

$$
\begin{aligned}
p_\theta(\textbf{z}) &\sim \mathcal{N}(\textbf{0}, \textbf{I})~\text{Notice}~p_\theta(\textbf{z})~\text{is parameter-free}\\
p_\theta(\textbf{x} \mid \textbf{z}) &\sim \mathcal{N}(\theta_\mu, \theta_{\sigma^2}\textbf{I})\\
q_\phi(\textbf{z} \mid \textbf{x}) &\sim \mathcal{N}(\phi_\mu, \phi_{\sigma^2}\textbf{I})
\end{aligned}
$$

Using this distributions maximize:

$$
\begin{aligned}
L(\theta,\phi; \textbf{x}) &= \mathbb{E}_{q_\phi (\textbf{z} \mid \textbf{x})} [ \log p_{\theta}(\textbf{x} \mid \textbf{z})] - D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert p_{\theta}(\textbf{z}))\\
&= \frac{1}{L} \sum_{l=1}^L \log p_{\theta}(\textbf{x} \mid \textbf{z}^{(l)})) - D_{KL}(q_\phi(\textbf{z} \mid \textbf{x}) \Vert p_{\theta}(\textbf{z}))\\
&= \frac{1}{L} \sum_{l=1}^L \log p_{\theta}(\textbf{x} \mid \textbf{z}^{(l)})) + \frac{1}{2}\sum_{j=1}^J(1+\log((\sigma^{(i)})^2)-(\mu^{(i)})^2-(\sigma^{(i)})^2) \\
\text{where}~ \textbf{z}^{(l)}&=g_\phi(\epsilon^{(l)}, \textbf{x})~\text{and}~\epsilon \sim p (\epsilon)
\end{aligned}
$$

I did promise some practical point of view to this: The corresponding objective function is: (You can find the full Pytorch code  \href{https://github.com/pytorch/examples/blob/master/vae/main.py}{here})

```python
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
```

As mentioned before, we can use MLP as the approximator to all our functions:

Encoder:


$$
\begin{aligned}
\log q_\phi(\textbf{z} \mid \textbf{x}) &= \log \mathcal{N}(\phi_\mu, \phi_{\sigma^2}\textbf{I)}\\
\text{where}\\
\phi_\mu &= \phi_{W_1}h + \phi_{b_1}\\
\log \phi_{\sigma^2} &= \phi_{W_2}h + \phi_{b_2}\\
\phi_h &= \tanh{\phi_{W_3}x+\phi_{b_3}}
\end{aligned}
$$

decoder:
$$
\begin{aligned}
\log p_\theta(\textbf{x} \mid \textbf{z}) &= \log \mathcal{N}(\theta_\mu, \theta_{\sigma^2}\textbf{I)}\\
\text{where}\\
\theta_\mu &= \theta_{W_1}h + \theta_{b_1}\\
\log \theta_{\sigma^2} &= \theta_{W_2}h + \theta_{b_2}\\
\theta_h &= \tanh{\theta_{W_3}z+\theta_{b_3}}
\end{aligned}
$$

So the corresponding Encoder and Decoder are:

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

# References

[1] [https://arxiv.org/pdf/1312.6114.pdf](https://arxiv.org/pdf/1312.6114.pdf)

[2] [http://gokererdogan.github.io/2017/08/15/variational-autoencoder-explained/](http://gokererdogan.github.io/2017/08/15/variational-autoencoder-explained/)

[3] [https://icml.cc/2012/papers/687.pdf](https://icml.cc/2012/papers/687.pdf)

[4] [https://www.youtube.com/watch?v=ogdv\_6dbvVQ](https://www.youtube.com/watch?v=ogdv\_6dbvVQ)

[5] [https://blog.evjang.com/2016/08/variational-bayes.html](https://blog.evjang.com/2016/08/variational-bayes.html)

[6] [https://xyang35.github.io/2017/04/14/variational-lower-bound/](https://xyang35.github.io/2017/04/14/variational-lower-bound/)

[7] [https://jaan.io/what-is-variational-autoencoder-vae-tutorial/](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)

[8] [http://andymiller.github.io/2016/12/19/elbo-gradient-estimators.html](http://andymiller.github.io/2016/12/19/elbo-gradient-estimators.html)

[9] [https://arxiv.org/pdf/1601.00670.pdf](https://arxiv.org/pdf/1601.00670.pdf)

[10] [https://github.com/pytorch/examples/blob/master/vae/main.py](https://github.com/pytorch/examples/blob/master/vae/main.py)

[11] [http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf](Pattern Recognition and Machine Learning, Bishop (2006))