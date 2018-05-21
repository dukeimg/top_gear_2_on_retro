oldspeed = 0
oldposition = 1
function reward ()
  newreward = 0
  newspeed = data.speed
  newpostion = data.position

  if newspeed > 1000 then
    if newspeed > oldspeed then
      x = 500
    else
      x = 1000
    end
    speedreward = (newspeed - oldspeed) / x
  else
    speedreward = -1
  end
  newreward = newreward + speedreward

  --Slowdown penalty
  if (oldspeed - newspeed) / oldspeed > 0.3 then
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
