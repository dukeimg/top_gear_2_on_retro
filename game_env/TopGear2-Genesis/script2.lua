oldposition = 20
oldprogress = 124
function reward ()
  newreward = 0
  progress = data.progress
  newpostion = data.position

  newreward = progress - oldprogress

  newreward = newreward + ((oldposition - newpostion) * 1000)

  oldposition = newpostion
  oldprogress = progress

  return newreward
end
