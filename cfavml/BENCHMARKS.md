# Benchmarks

This is a rough set of benchmarks of all the routines and their ballpark behaviour across various
CPUs and architectures on a _single core_.

All the servers are provided by Hetzner.

- [AMD Epyc (Zen 3)](#benchmarks---amd-epyc-zen-3)
- [Intel Xeon (Skylake)](#benchmarks---intel-xeon-skylake)
- [Ampere (ARM)](#bencharmks---ampere-arm)

The framework used for benchmarking is [Divan](https://github.com/nvzqz/divan), each benchmark
only uses one thread at a time.

## Disclaimer!

My servers and my CPUs are not your servers and your CPUs! You should take all these numbers and
their gains as a pinch of salt. If you want to truly find out if this library is useful to you or not
I _HIGHLY_ recommend you run your own tests on the hardware you expect to run.

These benchmarks mostly act as a way for me to sanity check my routines and make sure I haven't
broken things between changes.


## Benchmarks - AMD Epyc (Zen 3)

Ran on a Hetzner `CPX51 AMD x86`.

CPU Supports `AVX2`, `FMA` and the `SSE` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

#### CPU Info

```
processor	: 0
vendor_id	: AuthenticAMD
cpu family	: 23
model		: 49
model name	: AMD EPYC Processor
stepping	: 0
microcode	: 0x1000065
cpu MHz		: 2445.406
cache size	: 512 KB
physical id	: 0
siblings	: 16
core id		: 0
cpu cores	: 16
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm rep_good nopl cpuid extd_apicid tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext perfctr_core ssbd ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr wbnoinvd arat umip rdpid arch_capabilities
bugs		: sysret_ss_attrs null_seg spectre_v1 spectre_v2 spec_store_bypass retbleed smt_rsb srso
bogomips	: 4890.81
TLB size	: 1024 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 40 bits physical, 48 bits virtual
power management:
```

### Results

- [Run 2024-08-16](benchmark-runs/hetzner-cpx51-amd-cfavml-v0_2_0-2024-08-16.txt)


## Benchmarks - Intel Xeon (Skylake)

Ran on a Hetzner `CPX52 INTEL x86`.

CPU Supports `AVX512`, `AVX2`, `FMA` and the `SSE` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

#### CPU Info

```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 85
model name	: Intel Xeon Processor (Skylake, IBRS, no TSX)
stepping	: 4
microcode	: 0x1
cpu MHz		: 2099.998
cache size	: 16384 KB
physical id	: 0
siblings	: 16
core id		: 0
cpu cores	: 16
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault pti ssbd ibrs ibpb fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat pku ospke md_clear
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit mmio_stale_data retbleed gds bhi
bogomips	: 4199.99
clflush size	: 64
cache_alignment	: 64
address sizes	: 40 bits physical, 48 bits virtual
power management:
```

### Results

- [Run 2024-08-16](benchmark-runs/hetzner-cx52-intel-cfavml-v0_2_0-2024-08-16.txt)


## Bencharmks - Ampere (ARM)

Ran on a Hetzner `CAX41 Ampere ARM 64-bit`.

CPU Supports `NEON` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

### Results

- [Run 2024-08-16](benchmark-runs/hetzner-cax41-ampere-cfavml-v0_2_0-2024-08-16.txt)