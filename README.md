# TTC-GNN

Official implementation of **TTC-GNN (Textual–Topology Cross–Attentive Graph Neural Network)** for user engagement prediction in bot-centric social networks.

This repository contains the source code, data samples, and experimental results for the paper:

> **Jointly Modeling Persona-Conditioned Textual and Topological Cues for User Engagement Prediction in Bot-Centric Social Networks**

## Overview

LLM-driven social bots are increasingly deployed on social media platforms to participate in public conversations. However, user engagement with their posts is influenced not only by what bots say, but also by the social-network context in which they are embedded.

TTC-GNN jointly models **textual cues** and **topological cues** for engagement prediction. The framework consists of three main components:

- **Persona-Conditioned Textual Cue Encoder (PCE)**  
  Models post-level textual representations conditioned on bot persona descriptions and interpretable style prototypes.

- **Topological Cue Encoder (TCE)**  
  Uses GraphSAGE to encode structural and neighborhood information from the human–bot heterogeneous graph.

- **Dual Cross-Attention Module (DCA)**  
  Models bidirectional interactions between textual and topological representations before engagement prediction.

The task is formulated as a three-class classification problem with **low**, **medium**, and **high** engagement levels.

## Dataset

The dataset was collected from LLM-driven social bots deployed on a major microblogging platform.

The resulting dataset contains:

| Statistic | Value |
| --- | ---: |
| Social bots | 16 |
| Posts | 39,681 |
| Human nodes | 34,959 |
| Graph edges | 55,354 |
| Engagement classes | Low / Medium / High |

Posts are chronologically divided into training, validation, and test sets at an **80% / 10% / 10%** ratio for each bot.

Engagement labels are determined using empirical tertiles of comment counts in the training partition.

Due to platform policies and privacy considerations, this repository provides processed/sample data for reproducibility rather than redistributing unrestricted raw user data.

For detailed data organization, fields, and preprocessing instructions, please refer to:

(data_sample/README.md)
