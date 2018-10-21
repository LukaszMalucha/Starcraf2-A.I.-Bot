## TRAINING BOT

import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import *
import random
import cv2
import numpy as np
import time

HEADLESS = True

class Apollyon(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 170        
        self.MAX_WORKERS = 54
        self.choose_next_action = 0
        self.train_data = []

    def on_end(self, game_result):
        print('---end_game---')
        print(game_result)   


        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))    



    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_scv()
        await self.build_supplydepot()
        await self.lower_depot()
        await self.expand()
        await self.build_refinery()
        await self.build_barracks()
        await self.upgrade_barracks()       
        await self.build_factory()
        await self.upgrade_factory()  
        await self.build_armory()
        await self.build_starport()
        await self.build_defense_forces()
        await self.build_assault_forces()
        await self.game_map()
        await self.defend()
        await self.assault()
      

###################################################################### OPENCV VIZUALIZATION #################################################################################

    async def game_map(self):

        ## INITIAL IMAGE - NP ARRAY
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)  ## width by hight - reversed for array!, colors, datatype

        ## COLOR CODING    
        objects = {
                     COMMANDCENTER: [15, (0, 255, 0)],
                     SUPPLYDEPOT: [3, (50, 220, 30)],                    
                     REFINERY: [2, (55, 220, 30)],
                     BARRACKS: [3, (133, 178, 128)],

                     SCV: [1, (240, 240, 240)],
                     MARINE: [2, (50, 100, 218)],
                     MARAUDER: [2, (10 , 20, 50)],
                     MEDIVAC: [3, (130, 150, 200)],
                    }


        ## DRAW OBJECTS                
        for unit_type in objects:
            for unit in self.units(unit_type).ready:  
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), objects[unit_type][0], objects[unit_type][1], -1)  ## draw objects on a cv map


        ## DRAW ENEMY BASE/UNITS        
        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (240, 30, 30), -1)    ## small circle for other buildings
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (255, 0, 0), -1)      ## bigger circle for hq    

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (180, 70, 70), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 2, (255, 0, 0), -1)
 

        self.flipped = cv2.flip(game_data, 0)  ## flip the image            
        ## VIZUALIZATION FLIP
        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)  ## enlarge 2x 
            cv2.imshow('game_map', resized)
            cv2.waitKey(1)


####################################################################### GAME METHODS ##########################################################################



    async def build_scv(self):
        for center in self.units(COMMANDCENTER).ready.noqueue:
            if len(self.units(SCV)) < self.MAX_WORKERS:  
                if (self.units(SCV).amount - self.units(MARINE).amount) < 48:        
                    if self.can_afford(SCV):
                        await self.do(center.train(SCV))


    async def build_supplydepot(self):
        centers = self.units(COMMANDCENTER).first 
        if self.supply_used < 80:          
            if self.supply_left < 10 and not self.already_pending(SUPPLYDEPOT):
                await self.build(SUPPLYDEPOT, near=centers.position.towards(self.game_info.map_center, 4)) 
        else:
            if self.supply_left < 15 and self.supply_used < 190:
                await self.build(SUPPLYDEPOT, near=centers.position.towards(self.game_info.map_center, 4))
                      

    async def lower_depot(self):
        for depo in self.units(SUPPLYDEPOT).ready:
            await self.do(depo(MORPH_SUPPLYDEPOT_LOWER))
                       


    async def expand(self):
        if self.units(COMMANDCENTER).amount < 3 and self.can_afford(COMMANDCENTER):
            await self.expand_now()     



            
    async def build_refinery(self):
        if self.units(SUPPLYDEPOTLOWERED).exists:
            if self.vespene < 1000:
                for center in self.units(COMMANDCENTER).ready:            
                        vespenes = self.state.vespene_geyser.closer_than(20.0, center)
                        for vespene in vespenes:
                            if not self.can_afford(REFINERY):
                                break
                            worker = self.select_build_worker(vespene.position)
                            if worker is None:
                                break
                            if not self.units(REFINERY).closer_than(1.0, vespene).exists:
                                await self.do(worker.build(REFINERY, vespene))                  




## MILITARY BUILDINGS

    async def build_barracks(self):
        centers = self.units(COMMANDCENTER).first
        if self.units(SUPPLYDEPOTLOWERED).exists and self.units(COMMANDCENTER).amount > 1:
                if self.can_afford(BARRACKS) and self.units(BARRACKS).amount < 3:
                    if not self.already_pending(BARRACKS):
                        await self.build(BARRACKS, near=centers.position.towards(self.game_info.map_center, 7))


    async def upgrade_barracks(self):
        for barracks in self.units(BARRACKS).ready:
            if self.can_afford(BARRACKSTECHLAB) and not self.already_pending(BARRACKSTECHLAB):
                if barracks.add_on_tag == 0:
                    await self.do(barracks.build(BARRACKSTECHLAB))                           


    async def build_factory(self):
        centers = self.units(COMMANDCENTER).first
        if self.units(BARRACKS).exists:
            if self.units(FACTORY).amount < 3:
                if self.units(MARINE).amount > 2 and self.units(MARAUDER).amount > 2:
                    if self.can_afford(FACTORY):    
                        await self.build(FACTORY, near=centers.position.towards(self.game_info.map_center, 10))

    async def build_armory(self):
        centers = self.units(COMMANDCENTER).first
        if self.units(FACTORY).exists:
            if self.units(ARMORY).amount < 2 and not self.already_pending(ARMORY):
                if self.units(MARINE).amount > 5:
                    if self.can_afford(ARMORY):    
                        await self.build(ARMORY, near=centers.position.towards(self.game_info.map_center, 5)) 



    async def upgrade_factory(self):
        for factory in self.units(FACTORY).ready:
            if self.can_afford(FACTORYTECHLAB):
                if factory.add_on_tag == 0:
                    await self.do(factory.build(FACTORYTECHLAB))                    


    async def build_starport(self):
        if self.units(COMMANDCENTER).amount > 2:
            centers = self.units(COMMANDCENTER)[1] 
            if self.units(FACTORY).exists:
                if self.units(STARPORT).amount < 1 and not self.already_pending(STARPORT):
                    if self.units(MARINE).amount > 5:
                        if self.can_afford(STARPORT):    
                            await self.build(STARPORT, near=centers.position.towards(self.game_info.map_center, 10)) 
                                  

## MILITARY UNITS  


                    

    async def build_defense_forces(self):

        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE) and self.supply_left > 0:
                if self.units(MARINE).amount < 12 and self.can_afford(MARINE):
                    await self.do(barrack.train(MARINE))

        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARAUDER) and self.supply_left > 0:
                if self.units(MARAUDER).amount < 10 and self.can_afford(MARAUDER):
                    await self.do(barrack.train(MARAUDER))                         

                    


## COMPOSITION ~ let's Appollyon decide 

    async def build_assault_forces(self):                
        if self.supply_used < 190:
            if self.units(BARRACKS).ready.noqueue and self.units(FACTORY).ready.noqueue:
                choice = random.randrange(0, 4)
                target = False
                if self.iteration > self.choose_next_action:
                    if choice == 0:
                        for barrack in self.units(BARRACKS).ready.noqueue:
                            if self.can_afford(MARAUDER) and self.supply_left > 0:
                                if self.units(MARAUDER).amount < 30 and self.can_afford(MARAUDER):
                                    await self.do(barrack.train(MARAUDER))

                    if choice == 1:                
                        for factory in self.units(FACTORY).ready.noqueue:
                            if self.can_afford(CYCLONE) and self.supply_left > 0:
                                if self.units(CYCLONE).amount < 30 and self.can_afford(CYCLONE):
                                    await self.do(factory.train(CYCLONE))     

                    if choice == 2:                    
                        for factory in self.units(FACTORY).ready.noqueue:
                            if self.units(ARMORY).exists:
                                if self.can_afford(THOR) and self.supply_left > 0:
                                    if self.units(THOR).amount < 10 and self.can_afford(THOR):
                                        await self.do(factory.train(THOR))  

                    if choice == 3:                 
                        for starport in self.units(STARPORT).ready.noqueue:
                            if self.can_afford(MEDIVAC) and self.supply_left > 0:
                                if self.units(MEDIVAC).amount < 3 and self.can_afford(MEDIVAC):
                                    await self.do(starport.train(MEDIVAC))    


                    ## SAVE A CHOICE AS AN ARRAY [*,*,*,*]        
                    y = np.zeros(4)
                    y[choice] = 1
                    print(y)
                    self.train_data.append([y,self.flipped])
            



## ACTIONS

    def find_target(self, state):   
        if len(self.known_enemy_units) > 0:                     ## if there are any known units
            return random.choice(self.known_enemy_units)        ## return random one as a target
        elif len(self.known_enemy_structures) > 0:              ## ...or there any known structures
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]                ## ... otherwise go to enemy start location 



    async def defend(self):
        army = {MARINE: [10, 5],
                MARAUDER: [10, 2],
                CYCLONE: [10, 0],
                THOR: [10, 0],
                MEDIVAC: [5, 0],}           ## group units (attack/def)

        for UNIT in army:                     
            if self.units(UNIT).amount > army[UNIT][1]:                 ## Defend & gather until enough to attack
                if len(self.known_enemy_units) > 0:
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(random.choice(self.known_enemy_units)))  
    

    async def assault(self):

        target = self.enemy_start_locations[0]

        army = {MARINE: [10, 5],
                MARAUDER: [10, 3],
                CYCLONE: [10, 1],
                THOR: [10, 0],
                MEDIVAC: [5, 0],}           ## group units (attack/def)


        for UNIT in army:      
            if self.supply_used > 190:                 ## Defend & gather until enough to attack
                    for s in self.units(UNIT).idle:
                        await self.do(s.attack(target))        






run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Terran, Apollyon()),
    Computer(Race.Protoss, Difficulty.Medium)
    ], realtime=False)      
