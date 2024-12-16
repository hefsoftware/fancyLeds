import asyncio
import time
import struct
import random
import math
from bleak import BleakScanner, BleakClient

async def searchLedStrip():
    #devices = await BleakScanner.discover()
    #for d in devices:
    #    print(d)
  stop_event = asyncio.Event()
  connect_device=[None]
  def callback(device, advertising_data):
    #print("Callback", device, advertising_data)
    connect_device[0]=device
    stop_event.set()
  async with BleakScanner(callback, service_uuids=['6147c88b-8993-422c-97b6-32666c2b4109']) as scanner:
     await stop_event.wait()
     # Important! Wait for an event to trigger stop, otherwise scanner
     # will stop immediately.
     return connect_device[0]

def generatePacket(prog: int, fragment: int, offset: int, payload: bytearray):
  assert prog>=0 and prog<4
  assert fragment>=0 and fragment<128
  assert offset>=0 and offset+len(payload)<=2048
  
  ret=bytes([(prog<<3)|((offset>>8)&0x7), (offset&0xFF), fragment, 0])+payload
  return ret
class LedsProvider:
  def __init__(self):
    self.__totalBytes=0
  async def configuration(self, maxLeds):
    print("Max leds:", maxLeds)
    ret=tuple((x//2 for x in maxLeds))
    ret=(144*3,0,0)
    self.__totalBytes=sum(ret)
    print(f"Configuration for {maxLeds} will be {ret}")
    return ret
  async def data(self):
    nled=15
    return (b"\x00"*nled*3)+b"\xFF"*3+(b"\x00"*(self.__totalBytes-(nled+1)*3))
  
class LedStripHandler:
  def __init__(self):
    self.__client=None
    self.__datachar=None
    self.__notificationEvent=asyncio.Event()
    self.__numChunks=8
    self.__mtu=0
    self.__currentProg=0
    # Expected notified chunks
    self.__expected=[]
    # Characteristics
    self.__charConnectionParams=None
    self.__charData=None
    self.__charMaxLeds=None
    self.__charConfiguration=None
    # Chunks that are yet to be sent/were not notified in the proper slot
    self.__missed=[]
    self.__timeStart=0
    # Current transmission is finished
    self.__done=False
    self.__provider=None
  async def initialize(self, client, provider: LedsProvider):
    self.__client=client
    self.__provider=provider
    service=client.services['6147c88b-8993-422c-97b6-32666c2b4109']
    for ch in service.characteristics:
      if ch.uuid=='28190808-e588-44b3-a4b5-047465684ab9':
        self.__charConnectionParams=ch
      elif ch.uuid=='4f1a2462-abf8-4fda-9972-96dd17d39870':
        self.__charData=ch
      elif ch.uuid=='922604c8-ff53-40ef-8200-c08cbfa20324':
        self.__charMaxLeds=ch
      elif ch.uuid=='db1d3346-095d-4b9f-811e-7e0af1bd000c':
        self.__charConfiguration=ch
    print(self.__charData)
    connectionParams=await self.__client.read_gatt_char(self.__charConnectionParams)
    (interval, latency, self.__mtu)=struct.unpack("<HHH", connectionParams)
    print(f'Interval: {interval}, latency: {latency}, MTU:{self.__mtu}')
    maxLeds=await self.__client.read_gatt_char(self.__charMaxLeds)
    maxLeds=tuple((struct.unpack("<H", maxLeds[i:i+2])[0] for i in range(0, len(maxLeds), 2)))
    # TODO check out how many leds we actually want to use here
    configuredLeds=await provider.configuration(maxLeds)
    totalLeds=sum(configuredLeds)
    
    self.__numChunks=int(math.ceil(totalLeds/(self.__mtu-3)))
    print("About to configure for ",tuple(configuredLeds), totalLeds, self.__numChunks)
    await self.__client.write_gatt_char(self.__charConfiguration, struct.pack("<H", self.__numChunks)+b"".join((struct.pack("<H", x)) for x in configuredLeds), response=True)
    await self.__client.start_notify(self.__charData, self.__handleNotification)

  def __handleNotification(self,unused,payload):
    if len(payload)<1:
      return
    #print(payload,)
    payloadProg=payload[0]&3
    payloadNextProg=(payload[0]>>3)&3
    if payloadProg==self.__currentProg and payloadProg!=payloadNextProg:
      self.__done=True
      self.__currentProg=payloadNextProg
      self.__notificationEvent.set()
      return
    elif not self.__expected:
      return
    elif payloadProg==self.__currentProg:
      mask=payload[1:]
    else:
      mask=b''
    lastReceived=None
    received=set()
    # print("Received notification", mask, "expected", self.__expected)
    for (i,e) in enumerate(self.__expected):
      if e is None:
        continue
      if len(mask)>e//8 and mask[e//8]&(1<<(e%8)):
        # We received a chunk we were waiting
        # print("Received expected",e)
        received.add(e)
        lastReceived=i
    if lastReceived is not None:
      hadMissing=False
      for i in range(lastReceived+1):
        cur=self.__expected.pop(0)
        if cur is None:
          continue
        elif cur not in received:
          self.__missed.append(cur)
          hadMissing=True
      if hadMissing:
        self.__notificationEvent.set()
      elif not self.__expected:
        # Led controller received all chunks
        self.__notificationEvent.set()
    else:
      first=self.__expected.pop(0)
      if first is not None:
        # print("Missed: ", first)
        self.__missed.append(first)
        self.__notificationEvent.set()
    # print("- after ",time.time()-self.__timeStart,"expected ",self.__expected, "missed",self.__missed)
  async def sendSingleConfiguration(self, data):
    dataPerChunk=self.__mtu-7
    self.__missed=[i for i in range(self.__numChunks)]
    self.__timeStart=time.time()
    self.__done=False
    numSent=0
    while not self.__done:
      while self.__missed:
        curChunk=self.__missed.pop(0)
        # print(self.__missed, curChunk)
        sentProg=self.__currentProg
        if random.randint(0,10)<2 and False:
          # Randomly corrupts some chunks (so they will not be received)
          sentProg=(sentProg+2)&3
          # print("Corrupting chunk", curChunk, sentProg)
        out=generatePacket(sentProg, curChunk, curChunk*dataPerChunk, data[curChunk*dataPerChunk:(curChunk+1)*dataPerChunk])
        if not self.__expected:
          # Allows two notification slots before re-sending data
          self.__expected.append(None)
        self.__expected.append(curChunk)
        await self.__client.write_gatt_char(self.__charData, out, response=False)
        numSent=numSent+1
      await self.__notificationEvent.wait()
      self.__notificationEvent.clear()
    return (time.time()-self.__timeStart, numSent, numSent/self.__numChunks-1)
  async def loop(self):
    while True:
      t0=time.time()
      data=await self.__provider.data()
      t1=time.time()
      t=await self.sendSingleConfiguration(data)
      t2=time.time()
      print(f"Time to generate: {int((t1-t0)*1000)}ms, time to send: {int((t2-t1)*1000)}ms")
      #print("Transmit took", t)

async def main(provider):
  #print(generatePacket(False, 3, 280, b"Test"))
  print("Searching for LED strips")
  device=await searchLedStrip()
  print("Found: ", device)
  async with BleakClient(device) as client:
    print("Connected")        
    await asyncio.sleep(1)
    handler=LedStripHandler()
    await handler.initialize(client, provider)
    minTime=None
    maxTime=None
    totalTime=0.0
    n=0
    await handler.loop()
    #while True:
    #  t=await handler.sendSingleConfiguration()
    #  if minTime is None or minTime>t:
    #    minTime=t
    #  if maxTime is None or maxTime<t:
    #    maxTime=t
    #  totalTime+=t
    #  n+=1
    #  print(f"***** FINISHED IN {t:.3f} Min: {minTime:.3f} Max: {maxTime:.3f} Avg: {totalTime/n:.3f}")
    #print("About to disconnect")
    await client.disconnect()
if __name__=="__main__":
  asyncio.run(main(LedsProvider()))
