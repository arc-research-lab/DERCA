
<h2 align="center">
DERCA: Deterministic Cycle-Level Accelerator on Reconfigurable Platforms in DNN-Enabled Real-Time Safety-Critical Systems
</h2>

**DERCA is an accelerator architecture tailored for real-time safety critical systems.**

# Team
Principal Investigator: Prof. Peipei Zhou, https://peipeizhou-eecs.github.io/

Ph.D. Students: Shixin Ji (Student Lead), Zhuoping Yang, Xingzhen Chen, Wei Zhang, Jinming Zhuang

Faculty Collaborators: Prof. Alex Jones (Syracuse University), Prof. Zheng Dong(Wayne State University)

# 🚀 Thank You for Using DERCA!!
**Your support and growing engagement inspire us to continually improve and enhance DERCA**
- **Downloads since 1 Dec 2025:** <!--CLONES-->496<!--/CLONES-->
- **Views: since 1 Dec 2025:** <!--VIEWS-->231<!--/VIEWS-->
<p align="center">
  <picture>
    <img alt="DERCA" src="https://github.com/arc-research-lab/DERCA/blob/main/assets/DERCA_traffic_plot.png" width=90%>
  </picture>
</p>


# Quick Start guide
DERCA provides a commandline tool script to reproduce the figures shown in the manuscript:
```sh
#install required packages, virtual env or conda recommended
pip install -r requirements.txt 
#reproduce figures
#/DERCA# 
python reproduce_fig.py --target fig11a
python reproduce_fig.py --target fig11b
python reproduce_fig.py --target fig11c
python reproduce_fig.py --target fig13
```


# Introduction:
## DERCA Software stack
DERCA software stack takes (1) a task set and (2) the platform constraints as input, and generates the execution procedure of the taskset for execution.

the main steps of the DERCA software stack includes (1) parsing the input data, (2) enabling intra-layer preemption points, (3) conduct PP placement optimization and schedulability analysis, (4) simulation.

DERCA provides a commandline tool script to reproduce the figures shown in the manuscript:
```sh
#install required packages, virtual env or conda recommended
pip install -r requirements.txt 
#reproduce figures
#/DERCA# 
python reproduce_fig.py --target fig11a
python reproduce_fig.py --target fig11b
python reproduce_fig.py --target fig11c
python reproduce_fig.py --target fig13
```

For more detailed reproduce guide, please refer to the [artifact evaluation guide](artifact_evaluation_guide.md)

For more detailed explanation about the codes and workflow used in the software stack, please refer to the [DERCA software readme file](DERCA_SW/Readme.md)

## DERCA Hardware stack (under construction)
DERCA use [CHARM](https://github.com/arc-research-lab/CHARM) as the baseline accelerator, and we hand-tune the codes for our proposed imporvements.
The environment configuration of DERCA is the same as CHARM:
- Vitis: 2021.1
- Petalinux: xilinx-versal-common-v2021.1
- hardware platform: xilinx_vck190_base_202110_1.xpfm

Several artifacts representing each component and the whole system are/will be provided:
- [example 1](DERCA_HW/example1_recompute_dataflow/): implementation of recompute dataflow, controlled by CPU commands
- [example 2](DERCA_HW/example2_persist_dataflow/): implementation of persist dataflow, controlled by CPU commands
- example 3 (under construction): implementation of flexible dataflow, controlled by CPU commands
- example 4 (under construction): scheduler & kernel management module design with dummy accelerator and task release module
- example 5 (under construction): Final flexible accelerator design
