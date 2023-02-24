---
title: Is Apple rotten or not?
date: 2023-02-24 00:29:50 +0530
categories: [Fruits, Apple]
tags: [deep learning,fastai] 
---

# Is Apple rotten or not? 
Inspired by lesson 1 of Practical Deep Learning for Coders. In this project, we will be exploring the fascinating world of deep learning to identify and classify rotten apples. By leveraging the power of fastai library, we will create a simple model that can accurately distinguish between healthy and decayed apples. So, let's get started and unravel the mysteries of deep learning!

The basic steps we'll take are:
1. Use DuckDuckGo to search for images of "apple fruit photos"
2. Use DuckDuckGo to search for images of "rotten apple photos"
3. Fine-tune a pretrained neural network to recognise these two groups.
4. Try running this model on a picture of a apple and see if it works.


## Step 1: Installing and Importing the required libraries 


```python
!pip install -q duckduckgo_search
```


```python
from duckduckgo_search import ddg_images 
from fastcore.all  import *
```


```python
def search_images(term,max_images=30):
    print(f"Searching for :{term}")
    return L(ddg_images(term,max_results=max_images)).itemgot('image')
```


```python
urls=search_images('apple fruit photos')
urls[0]
```

    Searching for :apple fruit photos





    'https://wallpapercave.com/wp/wp2723042.jpg'




```python
from fastdownload import download_url 
dest='apple.jpg'
download_url(urls[0],dest,show_progress=False)
```




    Path('apple.jpg')




```python
from fastai.vision.all import * 
im=Image.open(dest)
im.to_thumb(256,256)
```




    
![png](https://user-images.githubusercontent.com/67200542/221076728-cf1bd673-7a39-477c-881e-7cc27e9bbfd1.png)
    




```python
download_url(search_images('rotten apple photos',max_images=1)[0],'rotten.jpg',show_progress=False)
Image.open('rotten.jpg').to_thumb(256,256)
```

    Searching for :rotten apple photos





    
![png](https://user-images.githubusercontent.com/67200542/221076721-251880bb-6c0c-47be-97c1-83aba6a4c969.png)
    



## Step 2: Creating dataset 


```python
searches='apple','rotten apple'
path = Path('fresh_or_not')

from time import sleep 

for o in searches:
    dest=(path/o)
    dest.mkdir(exist_ok=True,parents=True)
    download_images(dest,urls=search_images(f"{o} fruit photo"))
    sleep(10)
    resize_images(path/o ,max_size=400,dest=path/o)
```

    Searching for :apple fruit photo
    Searching for :rotten apple fruit photo


Removing the images which were not downloaded from the dataset


```python
failed=verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```




    1



## Step 3: Create Datablock

Fastai library provides an easy way to create model in the form of Datablock for more information click here [link](https://docs.fast.ai/data.block.html)


```python
dls=DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=parent_label,
    item_tfms=[Resize(192,method='squish')]
).dataloaders(path,bs=32)

dls.show_batch(max_n=6)
```


    
![png](https://user-images.githubusercontent.com/67200542/221076713-9d434254-2b3a-45c3-868a-805b6e0bb3e0.png)
    


## Step 4 : Fine tuning the model

```vision_learner ```
is provided from fastai.vision library for more info click here [link](https://docs.fast.ai/tutorial.vision.html)




```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

```

    /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.8/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.139429</td>
      <td>2.432094</td>
      <td>0.636364</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.582334</td>
      <td>1.200434</td>
      <td>0.454545</td>
      <td>00:19</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.359021</td>
      <td>0.563988</td>
      <td>0.181818</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.269847</td>
      <td>0.217871</td>
      <td>0.136364</td>
      <td>00:13</td>
    </tr>
  </tbody>
</table>


## Step 5 : Predictions


```python
is_fresh,_,probs = learn.predict('apple.jpg')
print(f"This is a: {is_fresh}.")
print(f"Probability it's a fresh apple: {probs[0]:.4f}")
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    This is a: apple.
    Probability it's a fresh apple: 0.9997


## Future scope and development. Here are a few possibilities:

1. Expansion to other fruits: The current project focuses on detecting the ripeness of apples. However, this technology can be expanded to other fruits as well. With additional training, the model could detect the freshness and ripeness of different types of fruits.

2. Real-time detection: The current project involves uploading an image to the model for analysis. However, with the development of IoT devices, it could be possible to develop a real-time detection system that could detect the ripeness of apples in real-time. This could be a significant boon for farmers and producers.

3. Integration with sorting machines: Currently, sorting machines rely on visual inspection by human operators to detect rotten fruits. However, with the development of deep learning models like the one in this project, it could be possible to integrate the model with sorting machines. This would automate the process of detecting rotten apples and increase efficiency.

3. Impact on food waste reduction: This project has the potential to reduce food waste by enabling early detection of rotten apples. With early detection, it will be possible to remove rotten apples before they contaminate the rest of the fruit in storage or transportation.

Overall, the "Is Apple Fruit Rotten?" project has a lot of potential for future development and could have a significant impact on the food industry.


