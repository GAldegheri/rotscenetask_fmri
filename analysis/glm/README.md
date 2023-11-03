The GLM analysis uses a series of **models** that each include specific conditions as regressors.

## Localizer models

1.	Faces, Objects, Scenes, Scrambled, button press
    
	**Contrasts:**
	1.	Object vs. Scrambled
	2.	Scenes vs. Objects
	3.	Faces vs. Scenes
	4.	Faces vs. Objects
	5.	Faces vs. Objects + Scenes
	6.	Scenes vs. Objects + Faces
	7.	Buttonpress vs. Presentation

2.	Stimulus, Baseline, button press
    
    **Contrasts:**
    1.	Stimulus vs. Baseline

3.	Object, Scrambled, Baseline, button press
    
    **Contrasts:**
	1.	Object + Scrambled vs. Baseline

___

## Training models


1.	Near (30°), Far (90°) + button press

	**Contrasts:**
	1.	Near vs. Far
	2.	Far vs. Near
	3.	Buttonpress vs. Presentation

2.	Wide, Narrow, button press

	**Contrasts:**
	1.	Wide vs. Narrow
	2.	Narrow vs. Wide

3.	Wide, Narrow (mini blocks), button press
4.	Miniblocks: Initial viewpoint, Final rotation (A30, B30, A90, B90), button press
5.	Same as 4, but decode A vs. B instead of 30 vs. 90
6.	Near, Far (mini blocks), button press
7.	A, B (mini blocks), button press

___

## Test models:
1.	Bed, Couch
2.	Near, Far
3.	Initial wide, Narrow

	**Contrasts:**
	1.	Initial wide vs. Narrow

4.	Final wide, Narrow

	**Contrasts:**
	1.	Final wide vs. Narrow
5.	Expected, Unexpected

	**Contrasts:**
	1.	Unexpected vs. Expected
	2.	Expected vs. Unexpected

6.	Final wide, Narrow, Expected, Unexpected

7.	Final wide, Narrow, Expected, Unexpected (randomly split in 3)

8.	Fake Final wide, Narrow, Expected, Unexpected (expected and unexpected assigned randomly)

9.	Expected (1, 2, 3), Unexpected

	**Contrasts:**
	1.	Unexpected vs. Expected

10.	Final wide, Narrow (including pre-stimulus interval and modeled as block)

11.	Final wide, Narrow, Expected, Unexpected (randomly split in 3, including pre-stimulus)

12.	Initial viewpoint, Final rotation, Expected (randomly split in 3), Unexpected

13.	Final rotation (30° vs 90°), Expected (split in 3), Unexpected

14.	Final rotation (30° vs. 90°), Expected (NOT split in 3), Unexpected

15.	Same as 12, but decode A30 vs. B30 and A90 vs. B90, not A30 vs. A90, etc.

16.	A30 vs. A90, B30 vs. B90, irrespective of exp/unexp

17.	A30 vs. B30, A90 vs. B90, irrespective of exp/unexp

18.	A vs. B (at probe onset - not initial view onset), irrespective of exp/unexp

19.	A vs. B (at probe onset - not initial view onset), expected (divided in 3), unexpected

20.	A30 vs. A90, B30 vs. B90, expected (divided in 3), unexpected (including pre-stimulus)

21.	Same as 20, but A vs. B

22.	A0, A30, A90, etc., exp. unexp. (Includes initial view too!)

23.	Same as 12, but with different random samples (and specified seed!)

24.	Same as 15 (A vs. B) with random samples from 23

25.	Final rotation (30° vs. 90°), synced to scene (not object) onset

26.	Final rotation (30° vs. 90°), with half of trials

27. Same as 9 (exp. 1, 2, 3, unexp.) but with specified seed.