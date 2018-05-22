oldspeed = 0
oldposition = 1
function reward ()
  newreward = 0
  newspeed = data.speed
  newpostion = data.position

  speedreward = newreward / 1000000.0
  newreward = newreward + speedreward

  --Slowdown penalty
  if oldspeed > 0 then
    if (oldspeed - newspeed) / oldspeed > 0.3 then
      newreward = newreward - 1000
    end
  end

  -- if car goes on the grass on the left side
  if data.side == 255 and data.pos < 35 or data.side == 254 then
    newreward = newreward - 100
  end

  -- if car goes on the grass on the right side
  if data.side == 0 and data.pos > 220 then
    newreward = newreward - 100
  end

  newreward = newreward + ((oldposition - newpostion) * 10000)

  oldposition = newpostion
  oldspeed = newspeed

  return newreward
end
