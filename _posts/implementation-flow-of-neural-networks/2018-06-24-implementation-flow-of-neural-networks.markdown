---
layout: post
thumbnail: "assets/implementation-flow-of-neural-networks/deconv_pic.png"
---


Looking back at the time I was a software engineer, when the program didn’t behave as I expected, usually the bug presented itself in all its glory. When I started writing deep neural networks this became less intuitive, a lot less.

While the research of neural networks is blooming, we still lack of thorough understanding of all its nuts and bolts. It doesn’t only hurt us from a theoretical point of view - Remember your new groundbreaking state-of-the-art model you want to submit to the next big conference, it needs to be implemented one way or the other, and if we are lacking that thorough understanding, this will make our job of implementing that model a lot more difficult.

The good news is that it is not that bad, while we are indeed still far away from complete understanding of neural networks, we do know quite a bit. Enough to know how to debug our model when it isn’t behaving as we expect, and enough to research the field and produce state-of-the-art models, in what seems like daily basis.

While there are some really great articles of checklists for how to build a good neural network (or fixing your current one) [3,4], trying every single suggestion from these lists is really time consuming and might not be in the right (gradient) direction. This is what this post is about, debugging neural networks — What to do when and why. **Go ahead and skip to the end if you want if-then like suggestions.**

<hr style="width:30%;margin:auto;margin-top:20px;margin-bottom:20px;height:2px;border:none;color:#333;background-color:#333;" />


For this I decided to present a use case of image segmentation on microscopy live cell images and talk about some of the problems you might encounter.

Always start by **looking at the data**, before suggesting a model.

> When you know the lyrics to a tune, you have some kind of insight as to it’s composition. If you don’t understand what it’s about, you’re depriving yourself of being really able to communicate this poem
Dexter Gordon

This quote obviously did not come from Yann LeCun, but from Dexter Gordon, but the thing is, you should be no different. Looking at the data will give you insights about the structure of it that you will not get by looking only on the loss graph.

![](/assets/implementation-flow-of-neural-networks/dataset.png)

Looking at the data, I spot two things quite quickly. The first, only 11 640x512 images are given, pretty small dataset. The second, while the segmentation is human made, the edges are not very smooth as you might expect, so perfect segmentation might be a bit out of reach.

Next thing you should do is read recent literature about similar problems as yours. While this goes without saying to people with research background, this is very important. I decided to implement UNet model [5]. My implementation is a refinement of the model here.

<script src="https://gist.github.com/mataney/cf0459589e833df5ec8b3bd8dd8baca8.js"></script>


Then I wrote a training paradigm. Notice I cropped each image to 128x128 image to fit our model (and the GPU memory).

<script src="https://gist.github.com/mataney/d5c0bb444fb0d3862ea3affff9ef40b8.js"></script>


Instantiate a model and train

{% highlight python %}
model = UNet(1, 3, 4, 0.1).to(device)
train_model(model)
{% endhighlight %}


![](/assets/implementation-flow-of-neural-networks/bug_in_loss_no_data_aug.png)
*<br>Training and Validation loss, first attempt*

Look closer at the y-axis marks, while it looks like a big drop in the beginning, the training loss is (almost) not decreasing at all through the training paradigm, this means our model is not capable to generalize. This indicates we either have a(t least one) bug in our model or the model doesn’t have the capacity of learning the task in hand, that is, it is too weak. (Obviously UNet is known to generalize well for the task of image segmentation so the model is strong enough, but here we want to imitate a scenario where we do not have any a-priori information about the model)

In order to find if it is a bug or a case of a weak model, we should try to **overfit our model on a small dataset**. While being able to overfit is not a sufficient condition for the model’s capacity of learning, it is still a necessary condition.

Let’s look at the training loss of a single example.

![](/assets/implementation-flow-of-neural-networks/bug_in_loss_no_data_aug_looking_on_a_single_image.png)
*<br>Training and Validation loss on a single example*

Even really weak models should be able to generalize better than what we see here, this looks a lot more like a bug.

Searching for the bug can be difficult. You should start by stripping your model and adding layer by layer to understand where the bug is, while you will not get the desired accuracy when your model is stripped, you expect to get lower loss with each component you add. (This advice is a bit more difficult to implement for NLP models such as seq2seq. IBM, Harvard NLP and CV groups have released this paper recently, I believe is a great tool and an important step towards a better way of debugging neural nets) 
One downfall I noticed when writing PyTorch code is view vs. transpose vs. permute (Even Andrej Karpathy agrees with me on this). Using one over the other might cause a “silent bug”, that is when matrices’ sizes are as expected but only when looking at the elements multiplied we see a mismatch. This is exactly what happened when I wrote the loss function causing wrong comparison between predictions and gold segmentation. Unless you have a better understanding to when to use view and transpose and permute than me, I believe you should give this step a minute and check if the output is what you expect. Let’s change the loss function to the following way:

{% highlight python %}
loss = criterion(pred, batch['seg'].squeeze(1))
{% endhighlight %}

![](/assets/implementation-flow-of-neural-networks/fix_bug_in_loss_looking_on_a_single_image.png)
*<br>Training and Validation loss on a single example —loss function fixed*


Finally we are able to overfit over a single image! Let’s run now on all our dataset. I’m adding a Jaccard accuracy evaluation to get extra information about our performance.

![](/assets/implementation-flow-of-neural-networks/looking_on_all_no_da_hidden_size_4.png)
*<br>Training and Validation loss and Jaccard — loss function fixed*


We are doing a much better job after fixing that bug. we got much better results on the train set, but we still overfit. While up until now we derived all our conclusion by looking at the loss graph, we can gain much better understanding of the model’s performance and what we should do in order make it better by **looking at examples where it failed and succeeded**.

![](/assets/implementation-flow-of-neural-networks/looking at examples.png)

Looking at our predicted segmentation we understand we are not doing that bad of a job, this means we are in the right direction, adding more layers and increasing hyperparameters might be beneficial here.


By increasing the hidden size of our UNet model we expect it to have a better representation of latent vectors, and by that a better generalization capability. While this is not always beneficial (I can think of a couple of times when the hidden size was too big for the model, and caused degeneration on other layers in the model), you should experiment with different hidden sizes.

{% highlight python %}
model = UNet(1, 3, 64, 0.1).to(device)
{% endhighlight %}

![](/assets/implementation-flow-of-neural-networks/looking_on_all_no_da_hidden_size_64.png)


While this is indeed a lot better it looks like we can get the scores even higher.

Notice in our first model we cropped the images to 128x128 size. While this is vital as our GPU might have difficulties processing the original images we are losing a lot of information taking only the images’ centers.

This leads us to do better image cropping and **data augmentation**. While some people tend to do image augmentation first, I believe dealing with smaller datasets at start will shorten the time between each of our working process iterations. Moreover, data augmentation sometimes might be expensive, time consuming or cause human suffering (e.g., medical tests), so sometimes data augmentation is the last resort.

<script src="https://gist.github.com/mataney/8caef039d0c5c1211f357cbed0d51b1f.js"></script>

![](/assets/implementation-flow-of-neural-networks/data_aug_bad_lr.png)

This is much better!

Last thing for now, what about the learning rate? I pretty much randomly set it to be 0.1, should I make it bigger? smaller? As Jeremy Howard mentioned in his brilliant fast.ai course [2], you should always use a learning rate finder, as suggested in the equally brilliant paper here (You know what, I’m not gonna compare brilliancy that easily, I will leave it to you to decide). There are some very good posts about this paper, a lot to do with fast.ai popularity, if you want to dive a bit deeper [8, 9].


{% highlight python %}
def find_lr(func, model, data_iter):
    min_lr = 0.00001
    max_lr = 10
    mul_by = (max_lr/min_lr)**(1/(len(data_iter)-1))
    lr = min_lr
    for i, batch in enumerate(data_iter):
        print(lr)
        for param_group in model.optim.param_groups:
            param_group['lr'] = lr
        lr *= mul_by
        loss, jac = func(i, model, batch)
        print("loss = " + str(loss))
{% endhighlight %}


I ran this function instead of the `run_proc_on_data` function for one epoch, resulting the following losses.

![](/assets/implementation-flow-of-neural-networks/learning_rate_finder.png)

In this graph I plotted both the learning rate and the corresponding loss at each timestep. We want to find the minimum of the loss curve, then multiply corresponding learning rate at about 1/10. resulting learning rates around 0.01–0.02. Running with `lr=0.01` we get the following performance

![](/assets/implementation-flow-of-neural-networks/data_aug_good_lr.png)

While this doesn’t look like much, validation set is reaching 85% Jaccard, quite a bit of an upgrade to the previous run. Remember you want to use some sort of early stopping as running for too long on the training set will tend to overfit.

## Conclusion


Looking at our segmentation now it looks like we are doing dramatically better. While we can do much more to try and perfect this score I think this is quite enough for one post.
We went through a case study of image segmentation and the steps you want to take. Yes, there is a lot more you can do, I didn’t talk about the Xavier initialization I used to control variance in results and the SGDR I implemented (a very good blog post about it here) but without much success.
While I presented steps you should take when presented with difficulties, I believe the world of debugging neural networks will develop and evolve. 
Exciting times ahead.


## TL;DR

Implementing flow for Neural Nets:
Start by Looking at a small sample of the data, know the data well before running your model.
Training loss is not acceptable?


1. Try overfitting on a small dataset:
If not working, look for a bug or understand why the model doesn’t have the capacity of learning and change accordingly
2. Have a look on the model’s successes and failures.
Change hyperparameters accordingly; try increasing your hyperparams sizes; change sizes according to the sizes you see in literature.
3. Use learning rate finder and some learning rate annealing technique.
4. Look again on the data, look at the training loss for small batch sizes, do we see any outliers, it might be too noisy or including wrong labels.

Training loss is acceptable but validation loss isn’t
1. Data augmentation! (If possible)
2. Regularization — L2 loss over parameter’s weights and Dropout. Batch normalization known to reduce overfitting as well.

## References

[1] [Deep Learning; Goodfellow, Bengio, Courville](http://www.deeplearningbook.org)

[2] [fast.ai](https://course.fast.ai)

[2] [https://medium.com/machine-learning-world/how-to-debug-neural-networks-manual-dc2a200f10f2](ttps://medium.com/machine-learning-world/how-to-debug-neural-networks-manual-dc2a200f10f2)

[3] [https://engineering.semantics3.com/debugging-neural-networks-a-checklist-ca52e11151ec](https://engineering.semantics3.com/debugging-neural-networks-a-checklist-ca52e11151ec)

[4] [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

[5] [SEQ2SEQ-VIS : A Visual Debugging Tool for Sequence-to-Sequence Models ](https://arxiv.org/pdf/1804.09299)

[6] [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

[7] [Kind-PyTorch-Tutorial](https://github.com/GunhoChoi/Kind-PyTorch-Tutorial)

[8] [https://miguel-data-sc.github.io/2017-11-05-first/](https://miguel-data-sc.github.io/2017-11-05-first/)

[9] [https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)