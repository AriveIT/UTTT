# UTTT bot

## Version 1:
minimax + ab pruning + position evaluation

### Stats:
Win/Loss/Tie vs Finn: 90 - 6 - 4

Dev Tool:
 - time elapsed: 0.07158109999727458
 - states evaluated: 2192

### Notes:
Position evaluation needs work
depth=4

## Version 2:
added transposition table

### Notes:
slowed things down, wasn't accessed enough
tt not good for uttt - going to leave tt out for now

## Version 3:
move ordering, based on not wanting to send opponent to center or to a full/won box

### Stats:
Win/Loss/Tie vs Finn: 93 - 2 - 5

Dev Tool:
 - time elapsed: 0.0626666999887675
 - states evaluated: 1901

### Notes:
consider somehow adding where sending opponent to position evaluation?
this may be evaluated implicitly
depth=4

## Version 4:
added man-advantages to position evaluation
i.e. incentivizes having man advantage in as many squares as possible (prioritizing mid, corn, side)

### Stats:
Win/Loss/Tie vs Finn: 94 / 0 / 6

Dev Tool:
 - time elapsed: 0.09114440018311143
 - states evaluated: 2217

## Version 5:
Messed around with evaluation fxn

### Stats:
Win/Loss/Tie vs Finn: 94 / 2 / 4

Dev Tool:
 - time elapsed: 0.09367650002241135
 - states evaluated: 2217

## Version 6:
Evaluated 0 / inf / -inf for win / ties

### Stats:
Win/Loss/Tie vs V5: 59 / 23 / 18

Dev Tool:
 - time elaped: 0.12643880024552345
 - states evaluated: 2217

## Version 7:
Clean up and refactoring

## Version 8:
Give bonus for winning boxes that are 2 in a rows

### Stats:
Win/Loss/Tie vs V7: 64 / 39 / 28

Dev Tool:
 - time elapsed: 0.1636206000111997
 - states evaluated: 2217

## Version 9:
Turn back on 2 in a row bonus for mini boxes

### Stats:
Win/Loss/Tie vs V8: 35 / 31 / 34

Dev Tool:
 - time elapsed: 0.4292958998121321
 - states evaluated: 2217

### Notes:
Seems to have minor benefit, extra time spent can probably be used more efficiently (very close to 0.5s limit)
Gonna turn back off for now

## Version 10:
Iterative deepening
Win/Loss/Tie vs V9: 8 / 1 / 1

### Notes:
Depth looks to be at least 6

## Version 11:
TT for move ordering for iterative deepening.

## Version 12:
Stopped throwing out last iteration
Win/Loss/Tie vs V11: 5 / 1 / 4

###Stats:
Win/Loss/Tie vs V10: 7 / 2 / 1




