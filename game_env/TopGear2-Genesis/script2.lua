oldposition = 20
oldprogress = 124
function reward ()
  newreward = 0
  progress = data.progress
  newpostion = data.position
  speed = data.speed

  newreward = progress - oldprogress

  speedreward = speed / 1000000.0
  newreward = newreward + speedreward

  newreward = newreward + ((oldposition - newpostion) * 1000)

  oldposition = newpostion
  oldprogress = progress

  return newreward
end
