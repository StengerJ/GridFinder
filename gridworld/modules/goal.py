import pygame as pg

from ._resources import image_path
from .block import Block

class Goal(pg.sprite.Sprite):
    def __init__(self,col,row):
        super().__init__()
        fpath=image_path("goal.png")
        self.image=pg.transform.scale(pg.image.load(fpath),Block.getBlockSize())
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*Block.sizeX
        self.rect.y=self.pos.y*Block.sizeY
        
    def draw(self, screen):
       screen.blit(self.image, (self.rect.x, self.rect.y))
