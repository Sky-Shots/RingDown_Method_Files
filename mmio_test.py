from periphery import MMIO

AXIL_BASE = 0x40000000
AXIL_SIZE = 0x1000

RAM_BASE  = 0x01000000
RAM_SIZE  = 4 * 1024 * 1024   # 4 MB ONLY

print("Mapping AXI...")
axil = MMIO(AXIL_BASE, AXIL_SIZE)
print("AXI OK")

print("Mapping RAM...")
ram = MMIO(RAM_BASE, RAM_SIZE)
print("RAM OK")

axil.close()
ram.close()

print("DONE")
