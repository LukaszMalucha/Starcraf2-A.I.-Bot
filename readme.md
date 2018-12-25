## STARCRAFT II A.I. Bot




### A.I. Apollyon

Apollyon is a mixture of role-based A.I. with an deep learning decision process enhancement. 
While most of it's actions are scripted, it decides about an army composition.


### Strategy 

#### 1. Survive Early Rush
Build a group of marines & marauders for early survival.

![1](https://user-images.githubusercontent.com/26208598/47260876-9d6ded00-d4bb-11e8-94bd-3d718231d34b.PNG)


#### 2. Expand Military Infrastructure 
Build & upgrade Barracks, Factories & Starports

![2](https://user-images.githubusercontent.com/26208598/47260877-9e9f1a00-d4bb-11e8-9608-2cb7fd3ec7d5.PNG)


#### 3. Build an Assault Force
Let the Apollyon decide among Marauders, Cyclones, Thors & Medivacs

![3](https://user-images.githubusercontent.com/26208598/47260879-a068dd80-d4bb-11e8-88a6-57346ada3412.PNG)

#### 4. Assault with full force
Attack after population reaches above 190.

![4](https://user-images.githubusercontent.com/26208598/47260880-a2cb3780-d4bb-11e8-853e-260ebd48d43d.PNG)



### Decision making process with Keras

If game ends with victory, random decisions made about army composition (which units to build) are stored as numpy array.
Once certain amount of games is beeing completed, data is beeing fed into deep neural network. Once model is trained it replaces random choice of action.


### Issues & Problems

It's not very clear how to position building on the map. Building addons can give alot of hard time. 
Building more complicated A.I. requires thousands of games played and a lot of processing power.



## Requirements:

1. Starcraft II Game - it's free now.
2. python-sc2 library:<br>
https://github.com/Dentosal/python-sc2
3. Maps Package:<br>
https://github.com/blizzard/s2client-proto#map-packs<br>
Create Maps folder in your Starcraft II directory then unzip it there (keep in folder packs)
4. Create folder "train_data"
5. Opencv | Keras | Tensorflow
6. Run .py file

## Credits:

#### Various Terran techniques:
[Sample Bots](https://github.com/Dentosal/python-sc2/tree/master/examples/terran)

#### Deep Learning Implementation:
[Python AI in StarCraft II Sentdex](https://www.youtube.com/watch?v=v3LJ6VvpfgI&list=PLQVvvaa0QuDcT3tPehHdisGMc8TInNqdq)


#### Game Guide:
[Terran Guide](https://liquipedia.net/starcraft2/Terran_Units_(Legacy_of_the_Void))

#### Useful tips & hacks:
[Discord SC2](https://discordapp.com/channels/350289306763657218/431774199753998346)


Thank you,

Lukasz Malucha