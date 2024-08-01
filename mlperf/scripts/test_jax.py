import jax
print("Before")
print(jax.local_devices())
for d in jax.local_devices():
  print(d.memory_stats())


for x in jax.live_arrays():
  x.delete()

jax.clear_caches()

print()
print()
print("After")
print(jax.local_devices())
for d in jax.local_devices():
  print(d.memory_stats())
