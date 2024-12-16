import LedDriver
import asyncio
import sys
import numpy
import time
import random
from scipy.spatial.transform import Rotation as R


def hslToRgb(h, s, l):
  h=float(h)%1.0
  s=float(s)
  l=float(l)
  def hueToRgb(p,q,t):
    if t<0: t+=1.0
    elif t>1: t-=1.0
    if t<1./6: return p+(q-p)*6*t
    elif t<1/2.: return q
    elif t<2/3.: return p+(q-p)*(2/3.-t)*6
    else: return p
  if s<=0:
    r=g=b=l
  else:
    q=l*(1-s) if l<0.5 else l+s-l*s
    p=2*l-q
    #p=q=1
    r=hueToRgb(p,q,h+1.0/3)
    g=hueToRgb(p,q,h)
    b=hueToRgb(p,q,h-1.0/3)
  return (r,g,b)

class LedEffect:
  def initialize(self, pos, oldColor):
    raise NotImplemented("Method not implemented")
  # Method should return an array of nleds*3 with RGB data (Range 0...1)
  def generate(self, time):
    raise NotImplemented("Method not implemented")
  def finished(self, time):
    return False
  def isInfinite(self):
    return True
def standardDecay(startColor, decay):
  return [c*(1-min(decay,1)) if c>=0.001 else 0.0 for c in startColor]

class CylinderEffect(LedEffect):
  def __init__(self, speed=160, colorFunction=None):
    self.__colorFunction=colorFunction if colorFunction is not None else  lambda rho, phi, y:hslToRgb(phi/(2*numpy.pi), rho, 0.05) 
    self.__speed=speed*2*numpy.pi/360
    self.__deltaPhi=0.0
    self.__oldTime=0
  def initialize(self, pos, oldColor):
    self.__pos=pos-numpy.average(pos, axis=(0,))
    self.__color=oldColor
    self.__maxR=max(map(lambda p: numpy.sqrt(p[0]**2+p[2]**2), self.__pos))
    y=pos[:,1]
    self.__ymin=numpy.min(y)
    self.__ymax=numpy.max(y)
  def generate(self, time):
    self.__deltaPhi=(self.__deltaPhi+(time-self.__oldTime)*self.__speed)%(numpy.pi*2)
    self.__oldTime=time
    for i in range(len(self.__pos)):
      #print(self.__pos[i])
      x,y,z,_=self.__pos[i]
      rho=numpy.sqrt(x**2+z**2)
      phi=(numpy.arctan2(z,x)+self.__deltaPhi)%(numpy.pi*2)
      self.__color[i]=self.__colorFunction(rho*2/self.__maxR, phi, y)
    return self.__color

class PlaneScrollEffect(LedEffect):
  def __init__(self, time=1.0, color=(0.2,0.0,0.0), decayTime=0.4, decayFunction=None):
    if decayFunction is None:
      decayFunction=standardDecay
    self.__color=color if callable(color) else lambda t: color
    self.__animTime=time
    self.__decayTime=decayTime
    self.__decayFunction=decayFunction
  def initialize(self, pos, oldColor):
    self.__pos=pos[:,1]
    self.__ledStartColor=numpy.copy(oldColor)
    self.__startDecayTime=numpy.zeros(len(self.__pos), numpy.single)
    self.__ledCol=numpy.copy(oldColor)
    self.__min=numpy.min(self.__pos)
    self.__max=numpy.max(self.__pos)
    self.__oldY=self.__min
  # Time always start from zero
  def generate(self, time):
    percent=time/self.__animTime
    newY=percent*(self.__max-self.__min)+self.__min
    for i in range(len(self.__pos)):
      y=self.__pos[i]
      if y>=self.__oldY and y<newY:
        newColor=self.__color(percent)
        assert len(newColor)==3
        self.__ledStartColor[i,:]=numpy.array(newColor)
        self.__startDecayTime[i]=time
      self.__ledCol[i,:]=self.__decayFunction(self.__ledStartColor[i,:], (time-self.__startDecayTime[i])/self.__decayTime)
    self.__oldY=newY
    return self.__ledCol
  def finished(self, time):
    return time>self.__animTime+self.__decayTime
  def isInfinite(self):
    return False

class RotateRandom(LedEffect):
  def __init__(self, effect):
    self.__effect=effect
  def initialize(self, pos, oldColor):
    x,y,z=random.random()*360,random.random()*360,random.random()*360
    rotMatrix=numpy.pad(R.from_euler('xyz', (x,y,z), degrees=True).as_matrix(), (0,1))
    rotatedPos=numpy.dot(pos, rotMatrix)
    self.__effect.initialize(rotatedPos, oldColor)
  def generate(self, t):
    return self.__effect.generate(t)
  def finished(self, t):
    return self.__effect.finished(t)
class EffectSerializer(LedEffect):
  def __init__(self, nextEffect):
    self.__nextEffect=nextEffect
  # nextEffect is a function that should return the next effect to show
  def initialize(self, pos, oldColor):
    self.__pos=pos
    self.__oldColor=oldColor
    self.__curEffect=None
    self.__startTime=0
  def generate(self, time):
    if self.__curEffect is None:
      self.__curEffect=self.__nextEffect()
      self.__startTime=time
      self.__curEffect.initialize(self.__pos, self.__oldColor)
    curT=time-self.__startTime
    self.__oldColor=self.__curEffect.generate(curT)
    if self.__curEffect.finished(curT):
      self.__curEffect=None
    return self.__oldColor
  def isInfinite(self):
    return self.__effect.isInfinite()
class FadeInOut(LedEffect):
  def __init__(self, effect, time):
    self.__effect=effect
    self.__time=time
    self.__finished=False
  def initialize(self, pos, oldColor):
    self.__effect.initialize(pos, oldColor)
  def generate(self, time):
    return self.__effect.generate(time)
  def finished(self, time):
    return time>self.__time

class SumEffects(LedEffect):
  def __init__(self, effects):
    self.__effects=tuple(effects)
  def initialize(self, pos, oldColor):
    for e in self.__effects:
      e.initialize(pos, oldColor)
    self.__old=numpy.copy(oldColor)
  def generate(self, time):
    values=[]
    for e in self.__effects:
      if not e.finished(time):
        values.append(e.generate(time))
    if len(values)>=1:
      result=numpy.zeros((len(self.__old), 3), numpy.single)        
      for v in values:
        result=result+v
      self.__old=result
    #print(self.__old)
    return self.__old
  def finished(self, time):
    return all(map(lambda e: e.finished(time), self.__effects))
  def isInfinite(self, time):
    return any(map(lambda e: e.isInfinite(), self.__effects))
      
class FancyLedsProvider:
  def __init__(self, effect):
    leds=numpy.ndarray((0,4), numpy.single)
    f=open("leds.txt")
    for (i,l) in enumerate(f):
      x,y,z=map(float,l.split("\t"))
      leds=numpy.vstack((leds,(x,y,z,0.0)))
    self.__ledsPos=leds
    self.__numLeds=numpy.size(leds,0)
    self.__effect=effect
    self.__effect.initialize(self.__ledsPos, numpy.zeros((self.__numLeds,3), numpy.single))
    self.__startTime=None
  async def configuration(self, maxLeds):
    print("Max leds:", maxLeds)
    ret=tuple([maxLeds[0]]+[0]*(len(maxLeds)-1))
    self.__numLeds=maxLeds[0]//3
    assert(self.__numLeds==numpy.size(self.__ledsPos,0))
    return ret
  async def data(self):
    t=time.time()
    if self.__startTime is None: self.__startTime=t
    curT=t-self.__startTime
    if self.__effect is not None:
      colors=self.__effect.generate(curT)
      if self.__effect.finished(curT):
        print("Animation finished")
        self.__effect=None
        sys.exit(0)
    assert(colors.shape==(self.__numLeds,3))
    colors=numpy.clip(colors*255, 0.0, 255.0).astype(int)
    #print(colors)
    ret=b""
    for led in range(self.__numLeds):
      ret+=bytes((colors[led,2],colors[led,1],colors[led,0]))
    return ret
# Return a function that returns a random color for an input parameter in range (0...1)
def generateRandomColorPattern(luminosity):
  startHue=random.random()
  deltaHue=random.random()
  if random.randint(0,1):
    deltaHue=-deltaHue
  def function(t):
    if t>1.0: t=1.0
    return hslToRgb(startHue+deltaHue*t, 1.0, luminosity)
  return function
#effect=EffectSerializer(lambda:RotateRandom(PlaneScrollEffect(time=random.random()*1.5+1.0, color=generateRandomColorPattern(0.2))))
#effect=EffectSerializer(lambda:FadeInOut(RotateRandom(CylinderEffect()), random.random()*10+4))

#effect1=EffectSerializer(lambda:RotateRandom(PlaneScrollEffect(time=random.random()*1.5+1.0, color=generateRandomColorPattern(0.2))))
#effect2=EffectSerializer(lambda:RotateRandom(PlaneScrollEffect(time=random.random()*1.5+1.0, color=generateRandomColorPattern(0.2))))
def generateEffect():
  r=random.randint(0,30)
  if r>2:
    ret=RotateRandom(PlaneScrollEffect(time=random.random()*1.5+1.0, color=generateRandomColorPattern(0.2)))
  else:
    ret=FadeInOut(RotateRandom(CylinderEffect()), random.random()*10+4)
  return ret
#effect1=EffectSerializer(lambda:RotateRandom(PlaneScrollEffect(time=random.random()*1.5+1.0, color=generateRandomColorPattern(0.2))))
#effect2=EffectSerializer(lambda:RotateRandom(PlaneScrollEffect(time=random.random()*1.5+1.0, color=generateRandomColorPattern(0.2))))
#effect3=EffectSerializer(lambda:FadeInOut(RotateRandom(CylinderEffect()), random.random()*10+4))
##effect4=EffectSerializer(lambda:FadeInOut(RotateRandom(CylinderEffect()), random.random()*10+4))
effect=SumEffects([EffectSerializer(generateEffect), EffectSerializer(generateEffect)])
#effect=RotateRandom(CylinderEffect())
#effect=FadeInOut(RotateRandom(CylinderEffect()), random.random()*10+4)
asyncio.run(LedDriver.main(FancyLedsProvider(effect)))
