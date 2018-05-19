oldspeed = 0
oldposition = 1
function reward ()
  newreward = 0
  newspeed = data.speed
  newpostion = data.position

  speedreward = (newspeed - oldspeed) / 500.0
  newreward = newreward + speedreward

  if oldposition - newpostion == 1 then
    newreward = newreward + 10
  elseif oldposition - newpostion == -1 then
    newreward = newreward - 10
  end

  oldposition = newpostion
  oldspeed = newspeed

  return newreward
end
