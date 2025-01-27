This position paper draws analogies between decision-making agents and Large Language Models (LLMs), and argues that agents should be trained like LLMs to achieve more general, robust, and aligned behaviors. As a proof of concept, we investigate training an agent by following the modern LLM training pipeline of unsupervised pre-training, supervised fine-tuning and reinforcement learning from human feedback (RLHF) on the Xbox game <a href="https://www.bleedingedge.com/en">Bleeding Edge</a>. This page provides videos of our agent at each stage of alignment, demonstating the benefits and effectiveness of training agents like LLMs.

## Motivation

Large Language Models (LLMs) demonstrate impressively general capabilities resulting from large-scale pre-training. Training an agent with large-scale imitation learning provides an analogous approach to learning in complex 3D environments from high-dimensional visual information (pixels). However, agents trained to imitate large-scale behavior data do not always perform the desired behaviors when deployed.

<figure>
<video autoplay muted loop playsinline style="pos: left; width: 49%">
    <source src="assets/Base Model/Bad Navigation.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 49%">
    <source src="assets/Base Model/Navigates Off Track.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: left; width: 49%">
    <source src="assets/Base Model/Not Reaching Jumppad.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 49%">
    <source src="assets/Base Model/Shadow Boxing.mp4" type="video/mp4">
</video>
<figcaption><blockquote>A 103M parameter GPT-style transformer agent, trained with imitation learning on 1.12 years of Bleeding Edge human gameplay. Not all imitated behaviors are desirable!</blockquote></figcaption>
</figure>

In this work, we consider an illustrative example where the agent spawns on an island with three jumppads (the yellow ramps in the above videos). We would like our agent to navigate directly to the left (or right) jumppad. We see that our general imitation learning agent sometimes performs this behavior, but not reliably, and over 40% of the time fails to reach any jumppad at all.

<center><figure>
  <img src="assets/images/Base_jumppad_success.png" alt="Base Imitation Model Success Rate" style="width:80%">
  <figcaption style="width:70%"><blockquote>General imitation learning agent jumppad success rates.</blockquote></figcaption>
</figure></center>

We draw an analogy between the undesirable behaviors of our imitation learning agent and the unhelpful respones of unaligned LLMs. Unaligned LLMs (trained only with unsupervised pre-training) contain a lot of knowledge, but frequently produce unhelpful responses, and must be aligned with subsequent supervised pre-training and reinforcement learning from human feedback (RLHF) stages to make them useful. Analagously, while scaling up our model and data can provide improved gameplay knowledge and generality, it provides no means for the agent to distinguish between expert and novice behaviors (or more generally, desired and undesired behaviors).

<!-- <center><figure>
  <img src="assets/images/shoggoth.jpg" alt="Shoggoth with Smiley Face" style="width:80%">
  <figcaption style="width:100%"><blockquote>Artistic illustration of LLM alignment. Source: <a href="https://huyenchip.com/2023/05/02/rlhf.html">https://huyenchip.com/2023/05/02/rlhf</a></blockquote></figcaption>
</figure></center> -->

By following the [modern LLM alignment pipeline](https://huyenchip.com/2023/05/02/rlhf.html), we hope to align our base imitation model to reliably perform the desired behavior, and make it useful. More generally, this may include refining the abilities of the agent, to achieve different objectives, obtain different gameplay styles or personalities, or just to achieve more human-like behavior.

<center><figure>
  <img src="assets/images/instructgpt_plot.png" alt="InstructGPT Performance Ablation" style="width:80%">
  <figcaption><blockquote>Alignment improves perceived helpfulness across langauge model sizes (InstructGPT). Does the same apply to agents? Source: <a href="https://openai.com/index/instruction-following">https://openai.com/index/instruction-following</a></blockquote></figcaption>
</figure></center>

---

## Supervised Fine-Tuning

We begin by fine-tuning our base imitation agent on curated trajectories that travel directly to a jumppad.

<figure>
<video autoplay muted loop playsinline style="pos: right; width: 32%">
    <source src="assets/Fine-Tuned Model/Successful Left.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: left; width: 32%">
    <source src="assets/Fine-Tuned Model/Successful Middle.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 32%">
    <source src="assets/Fine-Tuned Model/Successful Right.mp4" type="video/mp4">
</video>
<figcaption><blockquote>Demonstration trajectories of an agent going to the left, middle and right jumppads.</blockquote></figcaption>
</figure>

We find that our fine-tuned agent has an increased success rate for reaching all three jumppads, now only failing to reach a jumppad around 10% of the time. However, the agent still does not have a preference for a particular jumppad, and reaches all three in roughly even proportions (as expected from the training data).

<center><figure>
  <img src="assets/images/Fine-tuned_jumppad_success.png" alt="Fine-Tuned Imitation Model Success Rate" style="width:80%">
  <figcaption style="width:80%"><blockquote>Fine-tuned imitation learning agent jumppad success rates.</blockquote></figcaption>
</figure></center>

It is now natural to wonder whether pre-training the agent was beneficial, or would training directly on the fine-tuning trajectories have been just as effective? To answer this question, we train an equivalent agent from scratch on the fine-tuning trajectories. Interestingly, we find that this agent does not perform as well as the pre-trained agent. By reviewing trajectories to compare behaviors, we observe that pre-training generally makes the agent more robust to going out of distribution of the fine-tuning trajectories, since the agent has additional information from pre-training on how to return to the distribution of desired trajectories. An example of this phenomenon is demonstrated below.

<figure>
<video autoplay muted loop playsinline style="pos: left; width:49%">
  <source src="assets/Fine-Tuned Model/Fine-Tuned Missing but Turning Around.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width:49%">
  <source src="assets/Fine-Tuned Only Model/Agent Missing.mp4" type="video/mp4">
</video>
<figcaption><blockquote>The general pre-trained agent (left) is more robust to going out-of-distribution than the fine-tuned only agent (right). This simple but illustrative example demonstrates the benefits of incorporating general behavior data with unsupervised pre-training.
<strong>(Refresh page to align left and right videos for best comparison)</strong></blockquote></figcaption>
</figure>

## Preference Modeling

Still following the LLM alignment pipeline, we now train a reward model to capture our preferences about the fine-tuned agent's behavior. In our paper we use synthetic preferences to investigate how performance scales with preference labels (a proxy for human labeling time). We find that initializing the reward model with the pre-trained agent allows the reward model to capture our preferences much more accurately, enabling strong performance with comparatively few preference labels. This again demonstrates that progress in LLMs can be beneficial in the context of agents.

<center><figure>
  <img src="assets/images/reward_model_performances.png" alt="Reward model performances." style="width:80%">
  <figcaption style="width:80%"><blockquote>Reward model test performances.</blockquote></figcaption>
</figure></center>

## Alignment with Reinforcement Learning

We can now align the agent with our preferences by further fine-tuning the agent online with reinforcement learning using the reward models. To improve online alignemnt efficiency, we take inspiration from <a href="https://arxiv.org/abs/2308.08998">Reinforced Self-Training (ReST) (Gulcehre et al. 2023)</a>, originally introduced for efficient LLM alignment, and first fine-tuning on the trajectories which are labelled with the greatest reward. We refer to this additional alignment step as *preference fine-tuning*.

We find that with this improved alignment procedure we are able to reliably align our agent within our limited compute budget to reach both the left and the right jumppads.

### Left Jumppad Alignment

<figure>
<video autoplay muted loop playsinline style="pos: left; width: 49%">
    <source src="assets/Aligned towards Left Jumppad/Left Example 1.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 49%">
    <source src="assets/Aligned towards Left Jumppad/Left Example 2.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: left; width: 49%">
    <source src="assets/Aligned towards Left Jumppad/Left Example 3.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 49%">
    <source src="assets/Aligned towards Left Jumppad/Left Example 4.mp4" type="video/mp4">
</video>
<figcaption><blockquote>The agent has been aligned to consistently reach the left jumppad.</blockquote></figcaption>
</figure>

<center><figure>
  <img src="assets/images/Fully RL Aligned Left_jumppad_success.png" alt="Left-Aligned Model Success Rate" style="width:80%">
  <figcaption style="width:80%"><blockquote>Left-aligned agent jumppad success rates.</blockquote></figcaption>
</figure></center>

### Right Jumppad Alignment

<figure>
<video autoplay muted loop playsinline style="pos: left; width: 49%">
    <source src="assets/Aligned towards Right Jumppad/Right Example 1.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 49%">
    <source src="assets/Aligned towards Right Jumppad/Right Example 2.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: left; width: 49%">
    <source src="assets/Aligned towards Right Jumppad/Right Example 3.mp4" type="video/mp4">
</video>
<video autoplay muted loop playsinline style="pos: right; width: 49%">
    <source src="assets/Aligned towards Right Jumppad/Right Example 4.mp4" type="video/mp4">
</video>
<figcaption><blockquote>Alternatively, the agent can be aligned to consistently reach the right jumppad.</blockquote></figcaption>
</figure>

<center><figure>
  <img src="assets/images/Fully RL Aligned Right_jumppad_success.png" alt="Right-Aligned Model Success Rate" style="width:80%">
  <figcaption style="width:80%"><blockquote>Right-aligned agent jumppad success rates.</blockquote></figcaption>
</figure></center>

## Training and Alignment Summary

A summary of our training procedure and a heatmap of agent trajectories at each stage of alignment are shown below.

<center><figure>
  <img src="assets/images/Figure1.png" alt="Alignment Procedure Overview" style="width:100%">
  <figcaption><blockquote>Overview of our procedure for aligning an agent.</blockquote></figcaption>
</figure></center>
<center><figure>
  <img src="assets/images/Heatmap.png" alt="Agent Trajectories Heatmap" style="width:100%">
  <figcaption><blockquote>Heatmap of our agent's trajectories at each stage of alignment.</blockquote></figcaption>
</figure></center>

## Conclusion

Our proof-of-concept demonstrates that the modern LLM training procedure can be used to train agents to reliably perform desired behaviors in complex environments. This provides evidence for our position that many of the recent developments in training LLMs could be applied to agents to achieve similarly impressive models. However, there are many more aspects to investigate, which we believe will open up many exciting avenues of research in the coming years. Therefore, we call for enhanced communication and collaboration between the LLM and decision-making agents communities to enable shared insights, and provide a path towards more general and reliable agents for real-world applications. For more information and insights, please check out our position paper!