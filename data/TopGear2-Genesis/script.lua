oldspeed = 0
oldposition = 1
function reward ()
  newreward = 0
  newspeed = data.speed
  newpostion = data.position

  --speedreward = (newspeed - oldspeed) / 500.0
  speedreward = (newspeed / 500) * (newspeed / 5000)
  newreward = newreward + speedreward

  --Slowdown penalty
  if (oldspeed - newspeed) / oldspeed > 0.1 then
    newreward = newreward - 1000
  end

  if oldposition - newpostion == 1 then
    newreward = newreward + 10000
  elseif oldposition - newpostion == -1 then
    newreward = newreward - 10000
  end

  oldposition = newpostion
  oldspeed = newspeed

  return newreward
end
