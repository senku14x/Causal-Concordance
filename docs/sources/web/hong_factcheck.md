Hello,

This is my first post on Lesswrong. Hope my contribution makes the world a better and safer place.

**Note:**

**1\. This post is 100% human-written.**

**2\. Full paper in preparation for ICLR 2027**

  

Anthropic's NLA(Natural Language Autoencoder) is mostly confabulated... until I fixed the reward.
=================================================================================================

The NLA(The Natural Language Autoencoder([Anthropic, May 2026](https://transformer-circuits.pub/2026/nla/)) is a huge upgrade for the standard of mechanistic interpretability tool, SAE(Sparse AutoEncoder). It uses two copies of target model: AV(verbalizer) and AR(Reconstructor). The AV takes one inner vector and writes down what the model is thinking right now in English, and the AR reads the output of AV and regenerates the input vector.

Original Recipe
---------------

The AV is trained with RL(after a warm start) which is governed by pretty simple reward function:

$$
reward = -NMSE
$$

Where $\text{NMSE}=\frac{\sum_{i=1}^{N}\Vert{}x_i-\hat{x}_i\Vert{}^2}{\sum_{i=1}^{N}\Vert{}x_i\Vert{}^2}$.

After the training, Anthropic evaluated their work with FVE(Fraction of Variance Explained). An FVE of 0 corresponds to the mean activation $\bar{h}_l$, while an FVE of 1 is perfect reconstruction. Anthropic's checkpoint reached FVE of 0.6 to 0.8. When I ran their public checkpoints under a clean held-out split, they scored 0.712. The infrastructure was real and it worked as advertised.

  

The NLA revealed that, if the natural language description carries enough information to rebuild the activation, then the description will be the key to the inner psyche of the LLMs. As a result, Anthropic's agents more effectively discovered root cause of the model's misalignment test model's behavior than SAE and blackbox baselines.

  

FVE never evaluates the factual aspect
--------------------------------------

However, the reward function never punishes the AV for its unfaithful claim. It only asks whether the reconstructor, the AR, can decode them. Anthropic flagged the concern in their paper, saying: "NLA explanations can contain claims about the target model’s input context that are verifiably false." [One Lesswrong post](https://www.lesswrong.com/posts/LQXWiF8PyJ5ojNsEv) also showed that a LLM never sees the activation at all, could invent a description from the text and reconstructs about as well as the real AV that did see it. **A good reconstruction score is not the proof that AV is interpreting the internal vectors.**

  

I measured it by myself with Qwen 3.6-27B local model. My run more than doubled reconstruction FVE from 0.185 to 0.413. However, LLM judge found that the about 90% of the AV's specific claims(name, number, quotations) are not supported by the actual input. Also, 10 LLM judges favored only 54.5%\[45.2%, 63.5%\] of the time when showed the before-RL and after-RL results. The soar of FVE did not help us to read model's mind.

  

How NLA actually works
----------------------

**Conclusion first: "NLA loves the detail".**

I questioned "Which words in the AV's description carries the most reconstruction information?", "Which words contribute most to the activation reconstruct?". To answer it, I took the trained AV's descriptions, rewrote them in two different ways and measured how much reconstruction survived through the frozen reconstructor. In this configuration, FVE shows how much information the rewrite has destroyed. The table averages over 400 samples:

| what gets replaced | $Mean \\space {\\Delta FVE}$ |
| --- | --- |
| the specifics: names, numbers, quotes, swapped for different enitities but the same category | −0.219 |
| the abstract content: topics, styles (specifics kept) | −0.072 |

The entity-swap keeps every abstract word and replaces only the named things. For example, in one sample, *Bing Maps* becomes *Stripe Payment*, *Redmond, WA* becomes *Austin.* Reconstruction on that sample collapses (−0.89). The abstract-swap does the reverse, keeping Bing Maps and Redmond, while rewriting the framing ("technical geospatial context" becomes "financial accounting scenario," "developer-oriented" becomes "professional/corporate-style"). Reconstruction barely moves (−0.003). The named entities are what the reconstructor is actually reading.

  

Remember, the specifics were 90% false. AV tends to say "James is alive" when the actual text was "Mike is alive". But those incorrect details were carrying the most of information to reconstruction the activation. It is surprising that the AR actually relies on these false named entities. This is mostly because the named entities are the highest-bandwidth tokens available.

I have tested this on the Anthropic's public checkpoints and found the asymmetry of $\Delta FVE$ is even stronger(-0.394 vs -0.103). So the confabulation of AV was holding the entire building to not collapse. We should keep those detail while making the incorrect to correct.

  

Fixing the reward
=================

$$
reward = -NMSE + 0.25 \cdot (g - batch\_mean(g))
$$

where $g$ is a fact checking score between 0 and 1. Find the named entities(names, numbers, etc) in a description. Search the regex from original tokens where the activation came from. The g is the percentage you actually find. The AV said about ***Jake*** and ***17*** and only ***Jake*** appears in the original tokens, so the $g$ is 0.5 in this case.

  

What it fixed
-------------

Over 800 further training steps:

| steps with the term | fact checking $g$ | reconstruction FVE |
| --- | --- | --- |
| 0 (Anthropic recipe, my baseline) | 0.124 | 0.4126 |
| 400 | 0.268 | 0.4172 |
| 800 | 0.417 | 0.4194 |

  

![download.png](https://res.cloudinary.com/lesswrong-2-0/image/upload/v1784794830/lexical_client_uploads/gsdtdu4sufusufnptvi8.png)

Figure 1: The fact checking reward $g$ and reconstruction during the training with the new reward. Both curves are measured on data the training never saw.

The fact checking reward $g$ soared from 0.124 to 0.417. Now it starts to speak the truth! While preserving the reconstruction capability.

  

I also evaluated with the same 10 LLM judges, this time they favored the model trained with new reward function 83.6% \[75.6%, 89.4%\] of the time.

  

The J-space
===========

The J-space research([Anthropic, July 2026](https://www.anthropic.com/research/global-workspace)) identifies a small verbalizable subspace in activation space, carrying 6 to 7% activation energy. I split my reconstruction metric exactly into the part inside that J-space and the part outside. Across the reconstruction training, the inside part was nearly saturated from the start while the outside part almost tripled. About 95% of training gains live outside the J-space.

  

Then I ran the same split on every checkpoint trained with the new reward. Surprisingly, the inside/outside split is almost identical(diff under 0.1%) even as the $g$ grew 3.4x. The reward fix changed which word to carry the information, but did not change the activation space where the reconstruction lives. This might hint that the faithfulness is a property of the text channel alone.

  

Caveat
======

1.  The researches were conducted on one model, Qwen 3.6-27B.
2.  $g$ reward only checks exact string matches.

  

Conclusion
==========

1.  Anthropic's NLA is great for interpreting an LLM, but named entities are mostly false in the interpretation.
2.  By adding the fact checking reward $g$ to the original reward function, the model learns to speak truthfully, while preserving the information later used to reconstruct the activation.
3.  This is consistent with the latest research: J-space.