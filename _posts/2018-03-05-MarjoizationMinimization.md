---
layout: post
title: Sparsity in Signal Reconstruction
subtitle: Majorization and Minimization Derivation
use_math: true
---
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

If you're not aware of the excellent edX MOOC course currently running from the Technion on applications of sparsity in signal restoration you should definitely take a [look](https://www.edx.org/course/sparse-representations-image-processing-israelx-236862-2x).  It's being taught by Professor Michael Elad, the mind behind the K-SVD algorithm.  This is an excellent opportunity to learn from the best.

In one of his early lectures he provides an overview of the Majorization-Minimization method for ISTA, where he defines the auxiliary function:

$$ Q(\alpha, \alpha_0) = \lambda \| \alpha \|_1 + \frac{1}{2} \|z -HD\alpha \|_2^2 + \frac{c}{2} \|\alpha - \alpha_0 \|^2_2 - \frac{1}{2} ||HD (\alpha- \alpha_0) \|^2_2   $$ 

And states that this can be simplified into :

$$Q(\alpha, \alpha_0)  = \lambda \| \alpha \|_1  + \frac{c}{2} \Big\|\alpha -  \Big\{ \alpha_0 +\frac{1}{c}(HD)^T (z-HD\alpha_0) \Big\} \Big\|_2^2 + \text{ Const.}$$

This is an important result, as the subtracted term in the $$\ell_2$$ norm is not a function of $\alpha$.  Therefore, we can compact the equation into the following form:

$$ Q(\alpha, \alpha_0)  = \lambda \| \alpha \|_1  + \frac{c}{2} \Big\|\alpha -  v_0  \Big\|_2^2 + \text{Const.} $$

$$ v_0 =    \alpha_0 +\frac{1}{c}(HD)^T (z-HD\alpha_0)  $$
 
The full derivation of this result is left to the students, but as many taking the class may not have had fresh exposure to linear algebra, I have decided to provide the full derivation here as an educational reference.

Let's consider expanding out the additional terms in the cost function, folding all of the original terms into J:

$$ Q(\alpha, \alpha_0) = J(\alpha) + \frac{c}{2} \|\alpha - \alpha_0 \|^2_2 - \frac{1}{2} ||HD (\alpha- \alpha_0) \|^2_2   $$ 

$$ Q(\alpha, \alpha_0) = J(\alpha) +  (\alpha - \alpha_0)^T \frac{c}{2}(\alpha - \alpha_0)- \frac{1}{2} (\alpha- \alpha_0)^TD^TH^T HD (\alpha- \alpha_0)   $$ 

Which factored yields:

$$ Q(\alpha, \alpha_0) = J(\alpha) +  \frac{1}{2}(\alpha - \alpha_0)^T (cI - D^TH^THD)(\alpha - \alpha_0) $$

Let's replace J with the original expression and expand out the $$\ell_2$$ norm:
$$ Q(\alpha, \alpha_0) = \lambda \| \alpha \|_1 + \frac{1}{2} \|z -HD\alpha \|_2^2 +  \frac{1}{2}(\alpha - \alpha_0)^T (cI - D^TH^THD)(\alpha - \alpha_0)  $$

$$ Q(\alpha, \alpha_0) = \lambda \| \alpha \|_1 + \frac{1}{2} 
(z -HD\alpha)^T(z-HD\alpha) +  
\frac{1}{2}(\alpha - \alpha_0)^T (cI-D^TH^THD)(\alpha - \alpha_0)  $$

$$ Q(\alpha, \alpha_0) = \lambda \| \alpha \|_1 + \frac{1}{2} 
(z^Tz -2z^THD\alpha + \alpha^TD^TH^THD\alpha) +
\frac{1}{2}(\alpha - \alpha_0)^T (cI-D^TH^THD)(\alpha - \alpha_0) $$

Where we have observed the following two terms are equivalent constants due to the structure of the problem:

$$
z^THD\alpha = \alpha^TD^TH^Tz
$$

Since:

$$
z=HD\alpha  \quad \textrm { and} \quad z^Tz = c
$$

Let's hold off considering the $$\ell_1$$ term and the fractions for simplicity in the following:

$$  
(z^Tz -2z^THD\alpha + \alpha^TD^TH^THD\alpha) +
(\alpha - \alpha_0)^T (cI-D^TH^THD)(\alpha - \alpha_0)  $$


$$  
z^Tz -2z^THD\alpha + \alpha^TD^TH^THD\alpha +
\alpha^Tc\alpha-
\alpha^TD^TH^THD\alpha -
\ldots$$
$$
\ldots \alpha^T(cI-D^TH^THD)\alpha_0-
\alpha_0^T(cI-D^TH^THD)\alpha+
\alpha_0^T(cI-D^TH^THD)\alpha_0 
  $$

$$  
z^Tz + \alpha_0^T(cI-D^TH^THD)\alpha_0 
+\alpha^Tc\alpha 
-2z^THD\alpha
-2\alpha^T_0(cI-D^TH^THD)\alpha$$

$$  
z^Tz + \alpha_0^T(cI-D^TH^THD)\alpha_0 
+\alpha^Tc\alpha 
-2z^THD\alpha
-2\alpha^T_0(cI-D^TH^THD)\alpha$$

$$  
z^Tz + \alpha_0^T(cI-D^TH^THD)\alpha_0 
+\alpha^Tc\alpha
-2(z^THD
+\alpha^T_0(cI-D^TH^THD))\alpha$$

Note that the first three terms do not depend on $$\alpha$$, and we can simplify them as a constant factor:

$$Q(\alpha, \alpha_0) =
\alpha^Tc\alpha
-2(z^THD
+\alpha^T_0(cI-D^TH^THD))\alpha
+\text{Const.}
$$

To give some intuition about the next few steps we're going to take, it would be advantageous from an optimization framework if we could redefine the form of this equation in terms of an $\ell_2$ norm.  Let's remind ourselves what an $$\ell_2$$ norm looks like when expanded:

$$\|x-b\|_2^2 = x^Tx - 2b^Tx +b^Tb
$$

We already have an equation with the modulus of our dependent vector subtracted by two times a constant times our dependent vector.  This is already pretty close!  To that end, let's define:

$$b^T = \frac{1}{c}(D^TH^Tz + (cI-D^TH^THD)\alpha_0)
$$

Then redefining our equation in terms of b:

$$Q(\alpha, \alpha_0) =c(
\alpha^T\alpha
-2b^T\alpha)
+\text{Const.}
$$
 And therefore we can redefine Q in terms of an $$\ell_2$$ norm and replace the fraction 1/2 we removed previously:

$$Q(\alpha, \alpha_0) =c(
\alpha^T\alpha
-2b^T\alpha)
+\text{Const.}
$$

Therefore:
$$Q(\alpha, \alpha_0) =\frac{c}{2}\|\alpha-b\|_2^2
-\frac{c}{2}b^Tb
+\text{Const.}
$$

Note that we are interested in the derivative of Q, and everything outside of the norm is a constant with respect to $$\alpha$$!  Folding these terms into the constant, replacing b, and returning the $$\ell_1$$ norm, we are left with the result from the class:

Therefore:

$$Q(\alpha, \alpha_0) =\lambda \| \alpha \|_1 +\frac{c}{2}\|\alpha-b\|_2^2
+\text{Const.}
$$
$$Q(\alpha, \alpha_0) =\lambda \| \alpha \|_1 +\frac{c}{2}\Big\|\alpha-\frac{1}{c}(D^TH^Tz + (cI - D^TH^THD))\alpha_0\Big\|_2^2
+\text{Const.}
$$
$$Q(\alpha, \alpha_0) =\lambda \| \alpha \|_1 +\frac{c}{2}\Big\|\alpha-(\alpha_0 + \frac{1}{c}D^TH^T(z-HD\alpha_0)\Big\|_2^2
+\text{Const.}
$$
$$Q(\alpha, \alpha_0) =\lambda \| \alpha \|_1 +\frac{c}{2}\Big\|\alpha-(\alpha_0 + \frac{1}{c}(HD)^T(z-HD\alpha_0)\Big\|_2^2
+\text{Const.}
$$

Next time I will demonstrate how to implement ISTA for image deblurring in Python.
