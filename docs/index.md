## Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation

**Paper[\[OpenReview\]](https://openreview.net/forum?id=NnuYZ1el24C&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions))[\[ArXiv\]](https://arxiv.org/abs/2206.11403))/Code[\[GitHub\]](https://github.com/martius-lab/cee-us)**

**Abstract**: It has been a long-standing dream to design artificial agents that explore their environment efficiently via intrinsic motivation, similar to how children perform curious free play. Despite recent advances in intrinsically motivated reinforcement learning (RL), sample-efficient exploration in object manipulation scenarios remains a significant challenge as most of the relevant information lies in the sparse agent-object and object-object interactions. In this paper, we propose to use structured world models to incorporate relational inductive biases in the control loop to achieve sample-efficient and interaction-rich exploration in compositional multi-object environments. By planning for future novelty inside structured world models, our method generates free-play behavior that starts to interact with objects early on and develops more complex behavior over time. Instead of using models only to compute intrinsic rewards, as commonly done, our method showcases that the self-reinforcing cycle between good models and good exploration also opens up another avenue: zero-shot generalization to downstream tasks via model-based planning. After the entirely intrinsic task-agnostic exploration phase, our method solves challenging downstream tasks such as stacking, flipping, pick & place, and throwing that generalizes to unseen numbers and arrangements of objects without any additional training.

### Intrinsic Phase of CEE-US: Free Play in the Construction Environment
<div style="width: 75%; margin: 0 auto;">
{% include youtubePlayer.html id="tfdBPG201QI" %}
    </div>
                                                                                

### Extrinsic Phase of CEE-US: Solving Downstream Tasks Zero-Shot in the Construction Environment
We showcase the zero-shot generalization of our method to challenging object manipulation tasks. Thanks to the combinatorial generalization capabilities of Graph Neural Networks, we can apply the learned world model to solve tasks with more or less than 4 objects, which is the number of blocks seen during free play time. 

<div class="twocolumn_wrapper" style="margin-bottom: 3em;">

<div class="twocolumn_left">
    <h4>Stacking:</h4>
{% include youtubePlayer.html id="UG5EbkPEong" %}
</div>

    <div class="twocolumn_right">
        <h4>Throwing:</h4>
{% include youtubePlayer.html id="Eq-W90O8P8E" %}
      </div>
</div>
<div class="twocolumn_wrapper">
  <div class="twocolumn_left">
      <h4>Flipping:</h4>
{% include youtubePlayer.html id="OU6QWxlJm-s" %}
    </div>

  <div class="twocolumn_right">
<h4>Pick and Place:</h4>
{% include youtubePlayer.html id="8nOPQ4Ucito" %}
  </div>
  
 </div>

### Intrinsic Phase of CEE-US: Free Play in the Playground Environment
<div style="width: 75%; margin: 0 auto;">
{% include youtubePlayer.html id="pVDdZh0Gh4w" %}
    </div>

### Extrinsic Phase of CEE-US: Solving Downstream Tasks Zero-Shot in the Playground Environment
                                                                                
<div class="twocolumn_wrapper" style="margin-bottom: 3em;">
    <div class="twocolumn_left">
        <h4>Pushing with 4 Objects (as seen in free play):</h4>
{% include youtubePlayer.html id="LKKMerH52lI" %}
    </div>

     <div class="twocolumn_right">
         <h4>Pushing with 5 Random Objects:</h4>
{% include youtubePlayer.html id="3yINcEVRtFg" %}
    </div>
    </div>

<div class="twocolumn_wrapper">
    <div class="twocolumn_left">
        <h4>Pushing with 3 Random Objects:</h4>
{% include youtubePlayer.html id="xCzQK_9-IXI" %}
    </div>
    <div class="twocolumn_right">
    </div>
    </div>

### Intrinsic Phase of CEE-US: Free Play in the RoboDesk Environment
<div style="width: 75%; margin: 0 auto;">
{% include youtubePlayer.html id="feOHtPVHoew" %}
    </div>
                        
### Extrinsic Phase of CEE-US: Solving Downstream Tasks Zero-Shot in the RoboDesk Environment

We showcase the zero-shot generalization of our method in the Robodesk environment on the tasks: open sliding cabinet, push flat block off the table, open drawer and push green button.

<div class="twocolumn_wrapper" style="margin-bottom: 3em;">

<div class="twocolumn_left">
    <h4>Open Slide:</h4>
{% include youtubePlayer.html id="cuNqUZ5bC1w" %}
</div>

    <div class="twocolumn_right">
        <h4>Push Flat Block Off Table:</h4>
{% include youtubePlayer.html id="MSGgbomi_EU" %}
      </div>
</div>
<div class="twocolumn_wrapper">
  <div class="twocolumn_left">
      <h4>Open Drawer:</h4>
{% include youtubePlayer.html id="Unc3fqZvarU" %}
    </div>

  <div class="twocolumn_right">
<h4>Push Green Button:</h4>
{% include youtubePlayer.html id="K2kMPGlT75Y" %}
  </div>
  
 </div>
