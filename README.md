# SHG-VQA
Learning Situation Hyper-Graphs for Video Question Answering [CVPR 2023]

[Aisha Urooj Khan](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=ceiuCp4AAAAJ), [Hilde Kuehne](https://hildekuehne.github.io/), [Bo Wu](https://scholar.google.com/citations?user=6ozI_ZMAAAAJ&hl=en), Kim Chheu, Walid Bousselhum, [Chuang Gan](https://people.csail.mit.edu/ganchuang/), [Niels Da Vitoria Lobo](https://www.crcv.ucf.edu/person/niels-lobo/), [Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/)

[`Website`]() | [`Paper`](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/2023072364-4.pdf) | [`BibTeX`](#citation)

Official Pytorch implementation and pre-trained models for Learning Situation Hyper-Graphs for Video Question Answering (coming soon).

## Abstract
Answering questions about complex situations in videos requires not only capturing the presence of actors, objects,
and their relations but also the evolution of these relationships over time. A situation hyper-graph is a representation that describes situations as scene sub-graphs for video frames and hyper-edges for connected sub-graphs and has been proposed to capture all such information in a compact structured form. In this work, we propose an architecture for Video Question Answering (VQA) that enables answering
questions related to video content by predicting situation hyper-graphs, coined Situation Hyper-Graph based Video Question Answering (SHG-VQA). To this end, we train a situation hyper-graph decoder to implicitly identify graph representations with actions and object/human-object relationships from the input video clip. and to use cross-attention between the predicted situation hyper-graphs and the question embedding to predict the correct answer. The proposed method is trained in an end-to-end manner and optimized by
a VQA loss with the cross-entropy function and a Hungarian matching loss for the situation graph prediction. The effectiveness of the proposed architecture is extensively evaluated on two challenging benchmarks: AGQA and STAR. Our results show that learning the underlying situation hypergraphs helps the system to significantly improve its performance for novel challenges of video question-answering task.

<p align="center">
<img src="architecture.png" width=100% height=100% 
class="center">
</p>
