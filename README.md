# Codeup Capstone Project:  
We are building a predictive model on the relationship between oil recovery and the lateral length of wells without using complex geological parameters. We hypothesize that as the lateral length increases, the productivity per foot decreases in a non-linear manner. We further opine that horizontal well performance can be predicted from vertical wells.

### Contributors:  
> Eric Escalante  
> Gary Gonzenbach  
> Joseph Burton  
> Sandy Graham  


### Goals for the Project Are:
>- Scale oil recovery from vertical wells to horizontal wells to longer horizontal wells  
>- Finding a correlation between horizontal distance drilled and the amount of oil recovered   
>- Resample our wells by end date year and compare results to see if production has increased over time with the assumption of better technologies and processes

### Assumptions before we begin:
>- Cost increases per linear foot drilled  
>- Recovery per foot decreases as over a number of feet increases  
>- More risk with drilling horizontally; however it is a lot better than drilling new holes every time  
>- Geographic clustering of drilling data could lead us to different projections  
>- Higher proppant PPF/frac fluid theoretically leads to more recovery.  Not an assumption - higher costs as proppant PPF goes up  

### Data Dictionary:  
Attribute | Definition
------------ | -------------
API | American Petroleum Identification Code
Type | Vertical or Horizontal drill
MajorPhase | Which is the dominant substance (Gas, or Oil)
Formation |  Name of the layer of rock
Production Method | Which type of surgace maching is being used
Proppant Pound Per Foot | After injecting water for fracking, sand or ceramic is injected to hold open the layers of formation, to “prop” open the layers
Frac Fluid GPF | How much water is forced into the hold to frack it
Lateral Length | length of perforations to let the oil into the pipe
Frac Stages | Number of stages of fracking
Frac Fluid Type | Type of fluid used in the fracking process
First Prod | First production date
Last Prod | Last production date

