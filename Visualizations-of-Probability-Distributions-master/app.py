import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from math import sqrt


tfd = tfp.distributions
tfl = tfp.layers

st.title("Probability Distributions")
add_selectbox = st.sidebar.selectbox(
    'Choose an Option',
    ('Discrete Univariate', 'Continuous Univariate')
)



# st.title("1 dimensional normal distribution")

def Sum(p1val,p2val,zval):
    l = int(max(max(zval), len(zval)))
    l=l*2
    s=[0]*l
    for i in range(len(zval)):
        for j in range(len(zval)):
            s[int(zval[i]+zval[j])]+=p1val[i]*p2val[j]
    return s
def Product(p1val,p2val,zval):
    l=int(max(max(zval),len(zval)))
    l=l**2
    s=[0]*l
    for i in range(len(zval)):
        for j in range(len(zval)):
            s[int(zval[i]*zval[j])]+=p1val[i]*p2val[j]
    return s

def Normal():
    st.header("Normal distribution")
    p = tfd.Normal(2, 1)
    mean = st.slider('Mean', -5, 5, 0)
    std = st.slider('Scale', 0, 5, 1)
    z = f"""\\begin{{array}}{{cc}}
      \mu &= {mean} \\\\
      \sigma &= {std}
    \\end{{array}}
    """
    st.latex(z)
    st1 = r'''
        mean= \[\mu\]

Variance = \[ \sigma ^2\]

Entropy = \[ \frac{1}{2}\log (2 \pi \sigma ^2) + \frac{1}{2} \]
        '''

    st.latex(st1)
    q=tfd.Normal(mean,std)
    z_values = tf.linspace(-5, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'Distribution with unknown parameter', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'Distribution with given parameters')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Sum of two Normal Distributions")
    z_values1 = tf.linspace(-10, 10, 21)
    z_values1 = tf.cast(z_values1, tf.float32)

    mean1 = st.slider('Mean1', -5, 5, 0)
    std1 = st.slider('Std1', 0, 5, 1)
    mean2 = st.slider('Mean2', -5, 5, 0)
    std2 = st.slider('Std2', 0, 5, 1)
    q1=tfd.Normal(mean1,std1)
    q2=tfd.Normal(mean2,std2)
    prob_values_q1 = list(q1.prob(z_values1))
    prob_values_q2 = list(q2.prob(z_values1))

    fig2, (ax2,ax3) = plt.subplots(1,2)
    ax2.plot(z_values1, prob_values_q1, label=r'Normal(mean1,std1)')
    ax3.plot(z_values1, prob_values_q2, label=r'Normal(mean2,std2)')

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.set_title("Normal(mean1,std1)")
    ax2.set_ylim((0, 1))

    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.set_title("Normal(mean2,std2)")
    ax3.set_ylim((0, 1))

    st.pyplot(fig2)

    prob_values_sum = Sum(prob_values_q1, prob_values_q2, z_values1)
    q3 = tfd.Normal(mean1+mean2, sqrt(((std1)**2 + (std2)**2)))
    prob_values_q3 = q3.prob(range(len(prob_values_sum)))

    fig3, ax4 = plt.subplots()
    ax4.plot(range(len(prob_values_sum)), prob_values_sum, label=r'Normal(mean1,std1)+Normal(mean2,std2)', linestyle='--', lw=5, alpha=0.5)
    ax4.plot(range(len(prob_values_sum)), prob_values_q3, label=r'Normal(mean1+mean2, sqrt((std1^2 + std2^2)')

    ax4.set_xlabel("x")
    ax4.set_ylabel("PDF(x)")
    ax4.legend()
    ax4.set_ylim((0, 1))

    st.pyplot(fig3)
    st.markdown("Sum of two Normal distributions yields a Normal distribution")

    st.subheader("Relationship between Poisson and Normal Distribution")
    rate3 = st.slider('lambda1', 100, 500, 250, 50)
    q4 = tfd.Poisson(rate=rate3)
    z_values1 = tf.linspace(0, 600, 601)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_q4 = list(q4.prob(z_values1))
    q5 = tfd.Normal(rate3, sqrt(rate3))
    prob_values_q5 = list(q5.prob(z_values1))

    fig4, ax5 = plt.subplots()
    ax5.stem(z_values1, prob_values_q4, label=r'Poisson(lambda1)]', linefmt='r', markerfmt='ro')
    ax5.plot(z_values1, prob_values_q5, label=r'Normal(lamda1,sqrt(lambda1)', lw=3)
    ax5.set_xlabel("x")
    ax5.set_ylabel("PDF(x)")
    ax5.legend()
    ax5.set_ylim((0, 0.1))
    st.pyplot(fig4)
    st.markdown("For large values of lambda, Poisson(lambda) becomes approximately a normal distribution having mean= lambda and variance= lambda")

def Exponential():
    st.subheader("Exponential distribution")
    p = tfd.Exponential(2)
    rate = st.slider('Lambda', 1, 5, 1)
    cdf = r'''
            cdf  = $\int_a^b f(x)dx$ 
            '''

    st.latex(cdf)
    q = tfd.Exponential(rate)
    z_values = tf.linspace(-5, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Relationship between Gamma and Exponential Distribution")
    alpha1 = 1
    st.write("alpha1=1")
    beta1 = st.slider('beta1', 0.0, 10.0, 5.0, 0.5)
    q2 = tfd.Gamma(concentration=alpha1, rate=beta1)
    prob_values_q2 = list(q2.prob(z_values))
    q3 = tfd.Exponential(beta1)
    prob_values_q3 = list(q3.prob(z_values))
    fig3, ax3 = plt.subplots()
    ax3.plot(z_values, prob_values_q2, linestyle='--', lw=3, alpha=0.5, label=r'Gamma(1,beta)')
    ax3.plot(z_values, prob_values_q3, label=r'Exponential(beta)')

    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    # ax3.set_ylim((0, 1))
    st.pyplot(fig3)


def Uniform():
    st.subheader("Uniform distribution")
    p = tfd.Uniform(0,1)
    low = st.slider('low', 0, 5, 1)
    high = st.slider('high', 1, 6, 1)

    cdf = r'''
            cdf  = $\int_a^b f(x)dx$ 
            '''

    st.latex(cdf)
    q = tfd.Uniform(low,high)
    z_values = tf.linspace(-5, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")
    st.subheader("Relationship between Beta Distribution and Uniform Distribution")

    st.write("beta = alpha= 1")
    q4 = tfd.Beta(1, 1)
    q5 = tfd.Uniform(0, 1)
    z_values1 = tf.linspace(0, 1, 200)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_q4 = list(q4.prob(z_values1))
    prob_values_q5 = list(q5.prob(z_values1))

    fig3, ax3 = plt.subplots()
    ax3.plot(z_values1, prob_values_q4, label=r'Beta(1,1)', linestyle='--', lw=3)
    ax3.plot(z_values1, prob_values_q5, label=r'Uniform(0,1)')

    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    # ax3.set_ylim((0, 1))

    st.pyplot(fig3)


def Cauchy():
    st.subheader("Cauchy distribution")
    p = tfd.Cauchy(0, 0.5)
    loc = st.slider('location', 0.0, 5.0, 1.0, 0.5)
    sc = st.slider('scale', 0.0, 5.0, 1.0, 0.5)

    cdf = r'''
                        cdf  = $\int_a^b f(x)dx$ 
                        '''

    st.latex(cdf)
    q = tfd.Cauchy(loc, sc)
    z_values = tf.linspace(-5, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Relationship between Cauchy and StudentT Distribution")
    q2 = tfd.Cauchy(0,1)
    q3 = tfd.StudentT(1,0,1)
    prob_values_q2 = list(q2.prob(z_values))
    prob_values_q3 = list(q3.prob(z_values))

    fig2, ax2 = plt.subplots()
    ax2.plot(z_values, prob_values_q2, label=r'Cauchy(0,1)', linestyle='--', lw=3)
    ax2.plot(z_values, prob_values_q3, label=r'StudentT(1,0,1)')

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    ax2.set_ylim((0, 1))

    st.pyplot(fig2)


def Chi():
    st.subheader("Chi distribution")
    p = tfd.Chi(3)
    d = st.slider('dof', 0.0, 5.0, 1.0, 0.5)

    cdf = r'''
                        cdf  = $\int_a^b f(x)dx$ 
                        '''

    st.latex(cdf)
    q = tfd.Chi(d)
    z_values = tf.linspace(-5, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    #st.subheader("Transformation of Chi Distribution")
    #d2 = st.slider('dof2', 0, 5, 1, 1)
    #p1 = tfd.Chi(d2)
    #prob_values_new = list(p1.prob(z_values))
    #z_values2 = np.zeros(len(z_values))
    #for i in range(len(z_values2)):
     #   z_values2[i] = z_values[i]**2

    #q1 = tfd.Chi2(d2)
    #new = list(q1.prob(z_values))
   # fig2, ax2 = plt.subplots()
  #  ax2.plot(z_values, prob_values_new, label=r'X->Chi(d)', lw=3)
 #   ax2.plot(z_values2, prob_values_new, label=r'X transformed to X^2', lw=2)
#    ax2.plot(z_values2, new, label=r'Chi2(d)', linestyle='--', lw=2)

    #ax2.set_xlabel("x")
    #ax2.set_ylabel("PDF(x)")
    #ax2.legend()
    #st.pyplot(fig2)

def Chi_squared():
    st.subheader("Chi-squared distribution")
    p = tfd.Chi2(4)
    dof = st.slider('dof', 0.0, 10.0, 2.0,0.5)

    cdf = r'''
                        cdf  = $\int_a^b f(x)dx$ 
                        '''

    st.latex(cdf)
    q = tfd.Chi2(dof)
    z_values = tf.linspace(0, 10, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Relationship between Chi-Squared and Exponential Distribution")
    q2 =tfd.Chi2(2)
    q3 = tfd.Exponential(0.5)
    prob_values_q2 = list(q2.prob(z_values))
    prob_values_q3 = list(q3.prob(z_values))

    fig2, ax2 = plt.subplots()
    ax2.plot(z_values, prob_values_q2, label=r'Chi-Squared(2)', linestyle='--', lw=3)
    ax2.plot(z_values, prob_values_q3, label=r'Exponential(0.5)')

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    ax2.set_ylim((0, 1))

    st.pyplot(fig2)


def Laplace():
    st.subheader("Laplace distribution")
    p = tfd.Laplace(0, 3)
    m = st.slider('mu', 0, 5, 1)
    s = st.slider('sigma', 0, 5, 1)

    cdf = r'''
                        cdf  = $\int_a^b f(x)dx$ 
                        '''

    st.latex(cdf)
    q = tfd.Laplace(m, s)
    z_values = tf.linspace(-5, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")


def Pareto():
    st.subheader("Pareto distribution")
    p = tfd.Pareto(2, 1)
    a = st.slider('alpha', 0.0, 5.0, 2.5, 0.5)
    s = st.slider('scale', 0.0, 5.0, 2.5, 0.5)

    cdf = r'''
                        cdf  = $\int_a^b f(x)dx$ 
                        '''

    st.latex(cdf)
    q = tfd.Pareto(a, s)
    z_values = tf.linspace(0, 5, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=2, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

def Weibull():
    st.header("Weibull distribution")
    p = tfd.Weibull(1,2)
    k = st.slider('k', 0.0, 5.0, 0.5, 0.5)
    l = st.slider('lambda', 0.0, 5.0, 0.5, 0.5)

    cdf = r'''
                    cdf  = $\int_a^b f(x)dx$ 
                    '''

    st.latex(cdf)
    q = tfd.Weibull(k, l)
    z_values = tf.linspace(0, 10, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 2))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Relationship between Weibull and Exponential Distribution")
    l1 = st.slider('l', 0.0, 5.0, 1.0, 0.5)
    k1 = 1
    st.write("k=1")
    q2 = tfd.Weibull(k1, l1)
    prob_values_q2 = list(q2.prob(z_values))
    q3 = tfd.Exponential(1/l1)
    prob_values_q3 = list(q3.prob(z_values))
    fig3, ax3 = plt.subplots()
    ax3.plot(z_values, prob_values_q2, linestyle='--', lw=3, alpha=0.5, label=r'Weibull(1,l)')
    ax3.plot(z_values, prob_values_q3, label=r'Exponential(1/l)')

    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    ax3.set_ylim((0, 2))
    st.pyplot(fig3)

def StudentT():
    st.subheader("StudentT Distribution")
    p = tfd.StudentT(1,0,1)
    df = st.slider('df',0,5,2,1)
    loc = st.slider('loc', 0, 5, 2, 1)
    scale = st.slider('scale', 0, 5, 2, 1)

    cdf = r'''
                        cdf  = $\int_a^b f(x)dx$ 
                        '''

    st.latex(cdf)
    q = tfd.StudentT(df,loc,scale)
    z_values = tf.linspace(-10, 10 , 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=3, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)

    st.subheader("Relationship between Cauchy and StudentT Distribution")
    q2 = tfd.Cauchy(0, 1)
    q3 = tfd.StudentT(1, 0, 1)
    prob_values_q2 = list(q2.prob(z_values))
    prob_values_q3 = list(q3.prob(z_values))

    fig2, ax2 = plt.subplots()
    ax2.plot(z_values, prob_values_q2, label=r'Cauchy(0,1)', linestyle='--', lw=3)
    ax2.plot(z_values, prob_values_q3, label=r'StudentT(1,0,1)')

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    ax2.set_ylim((0, 1))

    st.pyplot(fig2)

def Beta():
    st.subheader("Beta distribution")
    p = tfd.Beta(1.5,1.1)
    alpha = st.slider('alpha', 0.0, 2.0, 1.2,0.1)
    beta = st.slider('beta', 0.0, 2.0, 1.2,0.1)

    cdf = r'''
                    cdf  = $\int_a^b f(x)dx$ 
                    '''

    st.latex(cdf)
    q = tfd.Beta(alpha, beta)
    z_values = tf.linspace(0, 1, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    #ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Relationship between Beta Distribution and Normal Distribution")
    alpha1 = st.slider('beta=alpha', 50, 500,250, 50)
    q2 = tfd.Beta(alpha1, alpha1)
    q3 = tfd.Normal(0.5,sqrt(0.25/(2*alpha1+1)))
    z_values1 = tf.linspace(0, 1, 200)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_q2 = q2.prob(z_values1)
    prob_values_q3 = q3.prob(z_values1)

    fig2, ax2 = plt.subplots()
    ax2.plot(z_values1, prob_values_q2, label=r'Beta(alpha,Beta)', linestyle='--', lw=3)
    ax2.plot(z_values1, prob_values_q3, label=r'Normal')

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    #ax2.set_ylim((0, 1))

    st.pyplot(fig2)

    st.subheader("Relationship between Beta Distribution and Uniform Distribution")

    st.write("beta = alpha= 1")
    q4 = tfd.Beta(1,1)
    q5 = tfd.Uniform(0,1)
    z_values1 = tf.linspace(0, 1, 200)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_q4 = list(q4.prob(z_values1))
    prob_values_q5 = list(q5.prob(z_values1))

    fig3, ax3 = plt.subplots()
    ax3.plot(z_values1, prob_values_q4, label=r'Beta(1,1)', linestyle='--', lw=3)
    ax3.plot(z_values1, prob_values_q5, label=r'Uniform(0,1)')

    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    #ax3.set_ylim((0, 1))

    st.pyplot(fig3)

    st.subheader("Transformation of Beta Distribution")
    alpha2 = st.slider('alpha2', 0.0, 2.0, 1.2,0.1)
    beta2 = st.slider('beta2', 0.0, 2.0, 1.8,0.1)
    p1 = tfd.Beta(alpha2, beta2)
    prob_values_new = list(p1.prob(z_values))
    z_values2 = np.zeros(len(z_values))
    for i in range(len(z_values2)):
        z_values2[i] = 1 - z_values[i]

    q1 = tfd.Beta(beta2,alpha2)
    new = list(q1.prob(z_values))
    fig2, ax2 = plt.subplots()
    ax2.plot(z_values, prob_values_new, label=r'X->Beta(alpha,beta)', lw=3)
    ax2.plot(z_values2, prob_values_new, label=r'X transformed to 1-X', lw=2)
    ax2.plot(z_values, new, label=r'X->Beta(beta,alpha)', linestyle='--', lw=2)

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    st.pyplot(fig2)


def Poisson():
    st.subheader("Poisson distribution")
    p = tfd.Poisson(5)
    rate = st.slider('lambda', 0, 10, 1)

    cdf = r'''
                cdf  = $\int_a^b f(x)dx$ 
                '''

    st.latex(cdf)
    q = tfd.Poisson(rate)
    z_values = tf.linspace(-2, 10, 13)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.stem(z_values, prob_values_p, label=r'Distribution with unknown parameters', linefmt='r', markerfmt='ro')
    ax.stem(z_values, prob_values_q, label=r'Distribution with given parameters', linefmt='--', markerfmt='bo')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))
    st.pyplot(fig)
    st.subheader("Addition of two Poisson distributions")
    rate1 = st.slider('lambda1', 0, 10, 1)
    rate2 = st.slider('lambda2', 0, 10, 1)
    q1 = tfd.Poisson(rate1)
    q2 = tfd.Poisson(rate2)
    prob_values_q1 = list(q1.prob(z_values))
    prob_values_q2 = list(q2.prob(z_values))
    fig2, (ax2,ax3) = plt.subplots(1,2)
    ax2.stem(z_values, prob_values_q1, linefmt='r', markerfmt='ro')
    ax3.stem(z_values, prob_values_q2, linefmt='--', markerfmt='bo')

    ax2.set_xlabel("x")
    ax2.set_title("Poisson(lambda1)")
    ax2.set_ylim((0, 1))

    ax3.set_xlabel("x")
    ax3.set_title("Poisson(lambda2)")
    ax3.set_ylim((0, 1))

    st.pyplot(fig2)

    prob_values_sum =Sum(prob_values_q1, prob_values_q2, z_values)
    q3 = tfd.Poisson(rate1+rate2)
    prob_values_q3 = list(q3.prob(range(len(prob_values_sum))))
    fig3, ax4 = plt.subplots()
    ax4.stem(range(len(prob_values_sum)), prob_values_sum, label=r'Poisson(lambda1)+Poisson(lambda2)', linefmt='r', markerfmt='ro')
    ax4.stem(range(len(prob_values_sum)), prob_values_q3, linefmt='--', label=r'Poisson(lambda1+lambda2)', markerfmt='bo')
    ax4.set_xlabel("x")
    ax4.set_ylabel("PDF(x)")
    ax4.legend()
    ax4.set_ylim((0, 1))
    st.pyplot(fig3)
    st.markdown("Summation of two poisson distribution with parameter lamba1 and lambda2 yields Poisson distribution with parameter (lambda1+lambda2)")

    st.subheader("Relationship between Poisson and Normal Distribution")
    rate3 = st.slider('lambda1', 100, 500,250,50)
    q4 = tfd.Poisson(rate=rate3)
    z_values1 = tf.linspace(0, 600, 601)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_q4 = list(q4.prob(z_values1))
    q5 = tfd.Normal(rate3, sqrt(rate3))
    prob_values_q5 = list(q5.prob(z_values1))

    fig4, ax5 = plt.subplots()
    ax5.stem(z_values1, prob_values_q4, label=r'Poisson(lambda1)]', linefmt='r', markerfmt='ro')
    ax5.plot(z_values1, prob_values_q5, label=r'Normal(lamda1,sqrt(lambda1)', lw=3)
    ax5.set_xlabel("x")
    ax5.set_ylabel("PDF(x)")
    ax5.legend()
    ax5.set_ylim((0, 0.1))
    st.pyplot(fig4)
    st.markdown("for large values of lambda, Poisson(lambda) becomes approximately a normal distribution having mean= lambda and variance= lambda")


def Binomial():
    st.subheader("Binomial distribution")
    p = tfd.Binomial(total_count=5, probs=.5)
    count = st.slider('n', 1, 10, 4, 1)
    prob = st.slider('prob', 0.0, 1.0, 0.5,0.1)
    st1 = r'''
                 Mean = \[n p \]
                Variance = \[n p q\]
                 Entropy = \[\frac{1}{2} \log_2 (2 \pi n e p q)\ + O ( \frac {1}{n}) \]
                '''

    st.latex(st1)
    q = tfd.Binomial(total_count=count, probs=prob )
    z_values = tf.linspace(0, 10, 11)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = list(p.prob(z_values))
    prob_values_q = list(q.prob(z_values))

    fig, ax = plt.subplots()
    ax.stem(z_values, prob_values_p, label=r'Distribution with unknown parameters', linefmt = 'r', markerfmt = 'ro')
    ax.stem(z_values, prob_values_q, label=r'Distribution with given parameters', linefmt ='--', markerfmt = 'bo')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)

    st.markdown("Relationship between Poisson and Binomial Distribution")
    count1 = st.slider('n', 750, 1000, 800, 25)
    prob1 = st.slider('prob', 0.0, 0.010, 0.001, 0.001)

    t = tfd.Binomial(total_count=count1, probs=prob1)
    r = tfd.Poisson(count1*prob1)

    z_values1 = tf.linspace(0, 50, 51)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_t = t.prob(z_values1)
    prob_values_r = r.prob(z_values1)

    fig1, ax = plt.subplots()
    ax.stem(z_values1, prob_values_t, label=r'binomial', linefmt='r', markerfmt='ro')
    ax.stem(z_values1, prob_values_r, label=r'poisson', linefmt='--', markerfmt='bo')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 0.5))

    st.pyplot(fig1)
    st.markdown("")
    st.markdown("It is clear from the graph, that for large values of n and small values of p, the binomial distribution approximates to poisson distribution.")
    st.markdown("")

    st.markdown("Transformation of Binomial Distribution")
    count2 = st.slider('n1', 1, 10, 4, 1)
    prob2 = st.slider('p', 0.0, 1.0, 0.5, 0.1)
    p1 = tfd.Binomial(total_count=count2, probs=prob2)
    prob_values_n_p = list(p1.prob(z_values))
    z_values2=np.zeros(len(z_values))
    for i in range(len(z_values2)):
        z_values2[i]=count2-z_values[i]

    q1 = tfd.Binomial(total_count=count2, probs=1-prob2)
    new = list(q1.prob(z_values))
    fig2, ax = plt.subplots()
    ax.stem(z_values, prob_values_n_p, label=r'X->Binomial(n,p)', linefmt='r', markerfmt='ro')
    ax.stem(z_values2, prob_values_n_p, label=r'X transformed to N-X', linefmt='g', markerfmt='go')
    ax.stem(z_values, new, label=r'X->Binomial(n,1-p)', linefmt='--', markerfmt='bo')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))
    ax.set_xlim((-0.5,count2+0.5))
    st.pyplot(fig2)
    st.markdown("When a r.v. X with the binomial(n,p) distribution is transformed to n-X, we get a binomial(n,1-p) distribution.")

    st.markdown("Relationship between Normal and Binomial Distribution")
    count3 = st.slider('n3', 750, 1000, 800, 25)
    prob3 = st.slider('prob3', 0.0, 1.0, 0.5, 0.1)

    t = tfd.Binomial(total_count=count3, probs=prob3)
    r = tfd.Normal(count3 * prob3, sqrt((count3*prob3*(1-prob3))))

    z_values1 = tf.linspace(0, 1000, 1001)
    z_values1 = tf.cast(z_values1, tf.float32)
    prob_values_t = t.prob(z_values1)
    prob_values_r = r.prob(z_values1)

    fig2, ax2 = plt.subplots()
    ax2.stem(z_values1, prob_values_t, label=r'Binomial', linefmt='r', markerfmt='ro')
    ax2.plot(z_values1, prob_values_r, label=r'Normal', lw=3)

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    ax2.set_ylim((0, 0.1))

    st.pyplot(fig2)
    st.markdown("")
    st.markdown("For large values of n, the binomial distribution approximates to the normal distribution with mean = n*p and variance = n*p*(1-p)")
    st.markdown("")


def Bernoulli_dist():
    st.header("Bernoulli distribution")
    p = tfd.Bernoulli(probs=0.5)
    suc = st.slider('p', 0.0, 1.0, 0.8, 0.1)

    pmf = r'''
                
                pmf  = f(x) = p  \quad  \qquad if x=1
                           \\ \qquad \qquad \quad \,\,\,= 1-p \quad   \,\, if x=0
                           \\ \,\,\,\quad\qquad= 0     \quad\qquad  else
                '''
    mean = suc
    variance = suc*(1-suc)

    st1 = f'''
            mean = p = {mean}\\\\
            variance = p*(1-p) = {variance:0.3f}\\\\
            Entropy = -q lnq - p ln p
            '''
    st.latex(pmf)
    st.latex(st1)
    q = tfd.Bernoulli(probs = suc)
    z_values = tf.linspace(0, 1, 2)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)
    fig, ax = plt.subplots()
    ax.stem(z_values, prob_values_p, label=r'Distribution with unknown parameters', linefmt='b', markerfmt='bo')
    ax.stem(z_values, prob_values_q, label=r'Distribution with given parameters', linefmt='g', markerfmt='go')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)
    st.subheader("Multiplication of two Bernoulli distributions")
    p1 = st.slider('p1', 0.0, 1.0, 0.2, 0.1)
    p2 = st.slider('p2', 0.0, 1.0, 0.7, 0.1)

    rv_p1 = tfd.Bernoulli(probs=p1)
    rv_p2 = tfd.Bernoulli(probs=p2)
    prob_values_p1 = rv_p1.prob(z_values)
    prob_values_p2 = rv_p2.prob(z_values)
    prob_values_pone = list(prob_values_p1)
    prob_values_ptwo = list(prob_values_p2)
    prob_values_pdt = Product(prob_values_pone, prob_values_ptwo, z_values)

    fig1,(ax1,ax2)=plt.subplots(1,2)
    ax1.stem(z_values, prob_values_p1, label=r'p1', linefmt='b', markerfmt='bo')
    ax2.stem(z_values, prob_values_p2, label=r'p2', linefmt='g', markerfmt='go')

    ax1.set_title("Bernoulli(p1)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("PDF(x)")
    ax1.set_ylim((0, 1))

    ax2.set_title("Bernoulli(p2)")
    ax2.set_xlabel("x")
    ax2.set_ylim((0, 1))

    st.pyplot(fig1)

    rv_p1p2 = tfd.Bernoulli(probs=p1*p2)
    prob_values_p1p2 = rv_p1p2.prob(range(len(prob_values_pdt)))

    fig2, ax3 = plt.subplots()
    ax3.stem(range(len(prob_values_pdt)), prob_values_pdt, label=r'Product of Bernoulli(p1) and Bernoulli(p2)', linefmt='r', markerfmt='ro')
    ax3.stem(range(len(prob_values_pdt)), prob_values_p1p2, label=r'Bernoulli(p1*p2)', linefmt='--', markerfmt='bo')
    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    ax3.set_ylim((0, 1))
    ax3.set_xlim((-0.5, 1.5))

    st.pyplot(fig2)
    st.markdown("Multiplication of a Bernoulli(p1) distribution with Bernouli(p2) gives a Bernoulli distribution with parameter=p1*p2")
    st.subheader("Addition of two Bernoulli distributions")
    p3 = st.slider('p3', 0.0, 1.0, 0.6, 0.1)
    rv_p3 = tfd.Bernoulli(probs=p3)
    prob_values_p3 = list(rv_p3.prob(z_values))
    prob_values_sum=Sum(prob_values_p3, prob_values_p3, z_values)
    fig3, ax4 = plt.subplots()
    ax4.stem(range(len(prob_values_sum)), prob_values_sum, label=r'Sum of 2 Bernoulli(p3) distributions', linefmt='r', markerfmt='ro')
    b = tfd.Binomial(total_count=2, probs=p3)
    prob_values_bin = list(b.prob(range(len(prob_values_sum))))
    ax4.stem(range(len(prob_values_sum)), prob_values_bin, label=r'Binomial(2,p3)', linefmt='--', markerfmt='bo')
    ax4.set_xlabel("x")
    ax4.set_ylabel("PDF(x)")
    ax4.legend()
    ax4.set_ylim((0, 1))

    st.pyplot(fig3)
    st.markdown("The sum of n Bernoulli(p) distributions is a binomial(n,p) distribution")

def BetaBinomial():
    st.markdown("Beta Binomial distribution")
    p = tfd.BetaBinomial(8,0.5,1.2)
    num = st.slider('n', 0, 10, 5, 1)
    al = st.slider('alpha', 0.0, 5.0, 0.2, 0.1)
    be = st.slider('beta', 0.0, 5.0, 0.2, 0.1)


    st1 = r'''
                Mean = \[ \frac {n \alpha}{\alpha + \beta} \]
                Variance =  \[ \frac {(n \alpha \beta)(\alpha + \beta + n)}{(\alpha + \beta + 1) (\alpha + \beta)^2} \]
                '''

    st.latex(st1)
    q = tfd.BetaBinomial(num, al, be)
    z_values = tf.linspace(0, 10, 11)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.stem(z_values, prob_values_p, label=r'Distribution with unknown parameters', linefmt='r', markerfmt='ro')
    ax.stem(z_values, prob_values_q, label=r'Distribution with given parameters', linefmt='--', markerfmt='bo')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)

    st.markdown("")
    st.markdown("Relarionship between BetaBinomial and Uniform-Discrete Distribution")
    st.markdown("")
    num2 = st.slider('n2', 0, 10, 5, 1)
    q2 = tfd.BetaBinomial(num2, 1, 1)
    prob_values_q2 = q2.prob(z_values)


    fig2, ax2 = plt.subplots()
    ax2.stem(z_values, prob_values_q2, label=r'BetaBinomial(n,1,1)', linefmt='r', markerfmt='ro')

    ax2.set_xlabel("x")
    ax2.set_ylabel("PDF(x)")
    ax2.legend()
    ax2.set_ylim((0, 1))

    st.pyplot(fig2)
    st.markdown("A BetaBinomial(n,alpha,beta) with alpha=beta=1 becomes a uniform-discrete distribution from 0 to n(n2 here)")
    st.markdown("")

    st.markdown("Relationship between Binomial and BetaBinomial Distribution")
    num3 = st.slider('_n', 0, 10, 5, 1)
    al2 = st.slider('_alpha', 100, 500, 200, 50)
    be2 = st.slider('_beta', 100, 500, 200, 50)

    q3 = tfd.BetaBinomial(num3, al2, be2)
    prob_values_q3 = q3.prob(z_values)

    success = al2/(al2+be2)
    q4 = tfd.Binomial(total_count=num3, probs=success)
    prob_values_q4 = q4.prob(z_values)

    fig3, ax3 = plt.subplots()
    ax3.stem(z_values, prob_values_q3, label=r'BetaBinomial', linefmt='r', markerfmt='ro')
    ax3.stem(z_values, prob_values_q4, label=r'Binomial', linefmt='--', markerfmt='bo')
    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    ax3.set_ylim((0, 1))

    st.pyplot(fig3)
    st.markdown("For large values of (alpha+beta), the BetaBinomial approximates to the Binomial Distribution with the probanility of success = alpha/(alpha+beta)")
    st.markdown("")

def Geometric():
    st.subheader("Geometric distribution")
    p = tfd.Geometric(probs=0.2)
    prob = st.slider('prob', 0.0, 1.0, 0.4, 0.1)

    st1 = r'''
                Mean=\[ \frac {1}{p} \]
                Variance= \[ \frac {1-p}{p^2} \]
                Entropy = \[ \frac { -(1-p) \log_2 {(1-p)} - p \log_2 p} {p}\]
                '''

    st.latex(st1)
    q = tfd.Geometric(probs=prob)
    z_values = tf.linspace(0, 10, 11)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.stem(z_values, prob_values_p, label=r'Distribution with unknown parameters', linefmt='b', markerfmt='bo')
    ax.stem(z_values, prob_values_q, label=r'Distribution with given parameters', linefmt='g', markerfmt='go')
    ax.set_xlabel("x")
    ax.set_ylabel("PMF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)

def Gamma():
    st.subheader("Gamma distribution")
    p = tfd.Gamma(concentration = 3.5, rate = 3 )
    concn = st.slider('concentration', 0.0, 10.0, 5.0, 0.5)
    rat = st.slider('rate', 0.0, 10.0, 2.0, 0.5)

    cdf = r'''
                cdf  = $\int_a^b f(x)dx$ 
                '''

    st.latex(cdf)
    q = tfd.Gamma(concentration = concn, rate = rat )
    z_values = tf.linspace(0, 20, 200)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.plot(z_values, prob_values_p, label=r'p', linestyle='--', lw=5, alpha=0.5)
    ax.plot(z_values, prob_values_q, label=r'q')

    ax.set_xlabel("x")
    ax.set_ylabel("PDF(x)")
    ax.legend()
    #ax.set_ylim((0, 1))

    st.pyplot(fig)
    kl = tfd.kl_divergence(q, p)
    st.latex(f"D_{{KL}}(q||p) \\text{{  is : }}{kl:0.2f}")

    st.subheader("Relationship between Gamma and Exponential Distribution")
    alpha1=1
    st.write("alpha1=1")
    beta1 = st.slider('beta1', 0.0, 10.0, 5.0, 0.5)
    q2 = tfd.Gamma(concentration=alpha1, rate=beta1)
    prob_values_q2 = list(q2.prob(z_values))
    q3 = tfd.Exponential(beta1)
    prob_values_q3 = list(q3.prob(z_values))
    fig3, ax3 = plt.subplots()
    ax3.plot(z_values, prob_values_q2, linestyle='--', lw=3, alpha=0.5, label=r'Gamma(1,beta)')
    ax3.plot(z_values, prob_values_q3, label=r'Exponential(beta)')

    ax3.set_xlabel("x")
    ax3.set_ylabel("PDF(x)")
    ax3.legend()
    #ax3.set_ylim((0, 1))
    st.pyplot(fig3)

def NegBin():
    st.markdown("Negative Binomial distribution")
    p = tfd.NegativeBinomial(total_count=5, probs=.5)
    count = st.slider('n', 1, 10, 1)
    prob = st.slider('prob', 0.0, 1.0, 0.1)
    cdf = r'''
                cdf  = $\int_a^b f(x)dx$ 
                '''

    st.latex(cdf)
    q = tfd.NegativeBinomial(total_count=count, probs=prob )
    z_values = tf.linspace(0, 100, 100)
    z_values = tf.cast(z_values, tf.float32)
    prob_values_p = p.prob(z_values)
    prob_values_q = q.prob(z_values)

    fig, ax = plt.subplots()
    ax.stem(z_values, prob_values_p, label=r'Distribution with unknown parameters', linefmt='b', markerfmt='bo')
    ax.stem(z_values, prob_values_q, label=r'Distribution with given parameters', linefmt='g', markerfmt='go')

    ax.set_xlabel("x")
    ax.set_ylabel("PMF(x)")
    ax.legend()
    ax.set_ylim((0, 1))

    st.pyplot(fig)



if (add_selectbox == "Continuous Univariate"):
    selection1 = st.sidebar.selectbox(
        'Choose an Option',
        ('Beta','Cauchy', 'Chi', 'Chi-Squared', 'Exponential', 'Gamma', 'Laplace', 'Normal',
         'Pareto', 'StudentT', 'Uniform', 'Weibull')
    )
    if (selection1 == "Normal"):
        Normal()
    elif (selection1 == "Exponential"):
        Exponential()
    elif(selection1 == "Uniform"):
        Uniform()
    elif(selection1 == "Error"):
        Error()
    elif (selection1 == "Cauchy"):
        Cauchy()
    elif (selection1 == "Chi"):
        Chi()
    elif (selection1 == "Chi-Squared"):
        Chi_squared()
    elif (selection1 == "Laplace"):
        Laplace()
    elif(selection1=="Pareto"):
        Pareto()
    elif(selection1 == "Weibull"):
        Weibull()
    elif (selection1 == "Gamma"):
        Gamma()
    elif (selection1 == "StudentT"):
        StudentT()
    else:
        Beta()
elif (add_selectbox == "Discrete Univariate"):
    selection1 = st.sidebar.selectbox(
        'Choose an Option',
        ('Bernoulli', 'Beta-Binomial','Binomial', 'Geometric', 'Negative-Binomial', 'Poisson')
    )
    if (selection1 == "Poisson"):
        Poisson()
    elif (selection1 == "Bernoulli"):
        Bernoulli_dist()
    elif (selection1 == "Binomial"):
        Binomial()
    elif (selection1 == "Beta-Binomial"):
        BetaBinomial()
    elif (selection1 == "Geometric"):
        Geometric()
    elif (selection1 == "Negative-Binomial"):
        NegBin()
